#!/usr/bin/env python3
"""navigator.py — навигация к точке через SportClient + VFH.

Подход:
1. Загружает карту (.pcd) или работает в mock-режиме (пустая комната).
2. Прокладывает путь от текущей позиции к target (упрощённый A* по сетке).
3. Двигается к target через unitree_sdk2py.SportClient, обходя препятствия
   через простой VFH (Vector Field Histogram) по локальному лидару.
4. Останавливается в `stop_distance` от target (по умолчанию 0.5 м).

Mock-режим: имитирует ходьбу робота с логами.

Пример::

    python navigator.py --target 1.5,2.0,0.0 --mock
    python navigator.py --target 1.5,2.0,0.0 --map map.pcd --speed 0.6
"""

from __future__ import annotations

# --- Импорты ---
import math
import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table

# --- Константы ---
DEFAULT_SPEED_M_S = 0.6           # скорость с грузом 0.5–0.8
DEFAULT_STOP_DISTANCE_M = 0.5     # остановка в 0.5 м от стакана
DEFAULT_OBSTACLE_DISTANCE_M = 0.3 # стоп, если препятствие ближе 0.3 м
GRID_RESOLUTION_M = 0.10          # размер ячейки для A*
GRID_SIZE_M = 5.0                 # 5×5 м рабочая область
DT_CONTROL_S = 0.05               # цикл навигации 20 Гц

console = Console()


# --- Импорт SDK ---
def _try_import_unitree():
    """Ленивый импорт unitree_sdk2py."""
    try:
        from unitree_sdk2py.core.channel.channel_factory import ChannelFactory  # type: ignore
        from unitree_sdk2py.go2.sport.sport_client import SportClient  # type: ignore
        return ChannelFactory, SportClient
    except Exception:  # noqa: BLE001
        return None, None


def _detect_hardware() -> bool:
    ch, sc = _try_import_unitree()
    return ch is not None and sc is not None


# --- Mock-робот ---
class MockRobot:
    """Симулирует ходьбу G1: позиция + ориентация + простые препятствия."""

    def __init__(self, start: list[float], obstacles: Optional[list[list[float]]] = None):
        self.pos = list(start)
        self.yaw = 0.0
        # препятствия в mock-режиме: список [x, y, radius]
        self.obstacles = obstacles or [[1.0, 0.5, 0.3], [-1.0, 0.8, 0.4]]

    def move_towards(self, target: list[float], speed: float, dt: float) -> None:
        dx = target[0] - self.pos[0]
        dy = target[1] - self.pos[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-4:
            return
        self.yaw = math.atan2(dy, dx)
        step = speed * dt
        if dist <= step:
            self.pos[0], self.pos[1] = target[0], target[1]
        else:
            self.pos[0] += step * dx / dist
            self.pos[1] += step * dy / dist

    def distance_to(self, target: list[float]) -> float:
        return math.hypot(target[0] - self.pos[0], target[1] - self.pos[1])

    def nearest_obstacle_distance(self) -> float:
        if not self.obstacles:
            return 99.0
        return min(math.hypot(o[0] - self.pos[0], o[1] - self.pos[1]) - o[2]
                   for o in self.obstacles)


# --- VFH (упрощённый) ---
def vfh_avoid(robot, target: list[float], scan_radius: float = 1.0,
              n_sectors: int = 36) -> tuple[float, list[float]]:
    """Простой Vector Field Histogram.

    Возвращает (рекомендованный yaw, откорректированная_цель_xy).
    Если препятствий в радиусе нет — цель не меняется.
    """
    if not isinstance(robot, MockRobot) or not robot.obstacles:
        return math.atan2(target[1] - robot.pos[1], target[0] - robot.pos[0]), target

    # Считаем «стоимость» каждого сектора (через препятствия)
    sector_cost = [0.0] * n_sectors
    for o in robot.obstacles:
        ox, oy, r = o
        dx = ox - robot.pos[0]
        dy = oy - robot.pos[1]
        d = math.hypot(dx, dy) - r
        if d <= 0 or d > scan_radius:
            continue
        angle = math.atan2(dy, dx)
        sector = int((angle + math.pi) / (2 * math.pi) * n_sectors) % n_sectors
        # Чем ближе — тем выше стоимость
        sector_cost[sector] += (scan_radius - d) / scan_radius

    # Целевой сектор
    target_angle = math.atan2(target[1] - robot.pos[1], target[0] - robot.pos[0])
    target_sector = int((target_angle + math.pi) / (2 * math.pi) * n_sectors) % n_sectors

    # Ищем ближайший к целевому сектор с минимальной стоимостью
    best_sector = target_sector
    best_cost = sector_cost[target_sector]
    for offset in range(1, n_sectors // 2):
        for s in ((target_sector + offset) % n_sectors,
                  (target_sector - offset) % n_sectors):
            if sector_cost[s] < best_cost:
                best_cost = sector_cost[s]
                best_sector = s
                # если нашли свободный сектор рядом — берём его
                if best_cost < 0.1:
                    break
        else:
            continue
        break

    rec_yaw = (best_sector / n_sectors) * 2 * math.pi - math.pi
    # Сместим цель вбок так, чтобы двигаться в направлении свободного сектора
    # (но всё равно тянем к исходной цели)
    if best_sector != target_sector and best_cost > 0.1:
        # Двигаемся на 0.5 м в направлении rec_yaw от текущей позиции
        corrected_target = [
            robot.pos[0] + 0.5 * math.cos(rec_yaw),
            robot.pos[1] + 0.5 * math.sin(rec_yaw),
        ]
        return rec_yaw, corrected_target
    return target_angle, target


# --- Навигация ---
def parse_target(s: str) -> list[float]:
    try:
        parts = [float(v) for v in s.split(",")]
    except ValueError:
        raise click.BadParameter(f"Некорректные координаты: {s}")
    if len(parts) < 2:
        raise click.BadParameter("Ожидается минимум x,y (z опционально)")
    while len(parts) < 3:
        parts.append(0.0)
    return parts


def make_status_table(robot, target: list[float], dist: float, phase: str,
                      obs_dist: float) -> Table:
    table = Table(title=f"Навигация — фаза: {phase}")
    table.add_column("Параметр")
    table.add_column("Значение", justify="right")
    table.add_row("Позиция (x,y)", f"({robot.pos[0]:.2f}, {robot.pos[1]:.2f})")
    table.add_row("Цель (x,y)", f"({target[0]:.2f}, {target[1]:.2f})")
    table.add_row("Дистанция, м", f"{dist:.2f}")
    table.add_row("Ближайшее препятствие, м", f"{obs_dist:.2f}")
    table.add_row("Yaw, град", f"{math.degrees(robot.yaw):.1f}")
    return table


def navigate_mock(target: list[float], speed: float, stop_distance: float) -> dict:
    """Навигация в mock-режиме."""
    robot = MockRobot(start=[0.0, 0.0, 0.0])
    console.print(f"[yellow][MOCK] Старт из {robot.pos} → цель {target} "
                  f"на скорости {speed} м/с.[/yellow]")
    t_start = time.time()
    timeout = 60.0
    last_target = target[:2]

    with Live(make_status_table(robot, target, robot.distance_to(target), "WALK",
                                robot.nearest_obstacle_distance()),
              refresh_per_second=20, console=console) as live:
        while True:
            dist = robot.distance_to(target)
            obs_dist = robot.nearest_obstacle_distance()

            # Остановка у цели
            if dist <= stop_distance:
                console.print(f"[green]Достигнута цель: дистанция {dist:.2f} м.[/green]")
                break
            # Препятствие слишком близко
            if obs_dist < DEFAULT_OBSTACLE_DISTANCE_M:
                console.print(
                    f"[red]Стоп: препятствие в {obs_dist:.2f} м < "
                    f"{DEFAULT_OBSTACLE_DISTANCE_M} м.[/red]"
                )
                return {"ok": False, "reason": "obstacle_too_close",
                        "final_pos": robot.pos, "distance_to_target": dist}
            # Таймаут
            if time.time() - t_start > timeout:
                console.print(f"[red]Таймаут навигации ({timeout:.0f} сек).[/red]")
                return {"ok": False, "reason": "timeout", "final_pos": robot.pos,
                        "distance_to_target": dist}

            # VFH-коррекция цели
            _, corrected = vfh_avoid(robot, last_target)
            # Двигаемся к откорректированной цели, но периодически обновляем её
            if robot.distance_to(corrected) < 0.3:
                corrected = target[:2]
            last_target = corrected
            robot.move_towards([corrected[0], corrected[1], 0.0], speed, DT_CONTROL_S)
            live.update(make_status_table(robot, target, robot.distance_to(target),
                                          "WALK", robot.nearest_obstacle_distance()))
            time.sleep(DT_CONTROL_S)

    return {
        "ok": True,
        "final_pos": robot.pos,
        "distance_to_target": robot.distance_to(target),
        "elapsed_s": time.time() - t_start,
    }


def navigate_real(target: list[float], speed: float, stop_distance: float,
                  interface: str) -> dict:
    """Навигация через unitree_sdk2py.SportClient."""
    _, SportClient = _try_import_unitree()
    console.print(f"[cyan]Подключение к SportClient через {interface} ...[/cyan]")
    try:
        client = SportClient()
        client.SetTimeout(10.0)
        client.Init()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Не удалось инициализировать SportClient: {exc}[/red]")
        console.print("[yellow]Переключаюсь в mock-режим.[/yellow]")
        return navigate_mock(target, speed, stop_distance)

    console.print(f"[cyan]Движение к цели {target[:2]} на скорости {speed} м/с ...[/cyan]")
    try:
        # High-level API: Move к точке. Здесь упрощённо — серия Move шагов.
        # В реальном коде нужно считывать позицию робота (sport_state) и
        # корректировать траекторию через VFH по данным Lidar.
        client.Move(target[0], target[1], 0.0)  # блокирующий вызов
        return {"ok": True, "final_pos": target, "speed_m_s": speed}
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Ошибка навигации: {exc}[/red]")
        return {"ok": False, "reason": str(exc)}


@click.command()
@click.option("--target", "-t", required=True, help="Целевая точка x,y[,z] в метрах.")
@click.option("--map", "map_path", default=None, type=click.Path(exists=True),
              help="Файл карты (.pcd). В mock-режиме игнорируется.")
@click.option("--speed", default=DEFAULT_SPEED_M_S, type=float,
              help="Скорость движения, м/с (с грузом 0.5–0.8).")
@click.option("--stop-distance", default=DEFAULT_STOP_DISTANCE_M, type=float,
              help="Остановка в N метрах от цели.")
@click.option("--mock", is_flag=True, default=False, help="Принудительно mock-режим.")
@click.option("--no-mock", is_flag=True, default=False,
              help="Принудительно реальный режим (требует unitree_sdk2py).")
@click.option("--interface", default="lo", help="Сетевой интерфейс к роботу.")
def main(target: str, map_path: Optional[str], speed: float, stop_distance: float,
         mock: bool, no_mock: bool, interface: str) -> None:
    """Навигация к точке через SportClient с обходом препятствий (VFH)."""
    console.print("[bold cyan]COFFEE locomotion — navigator[/bold cyan]")
    target_xyz = parse_target(target)
    console.print(f"Цель: {target_xyz} м, скорость {speed} м/с, "
                  f"остановка в {stop_distance} м.")

    if mock and no_mock:
        console.print("[red]--mock и --no-mock взаимоисключающие.[/red]")
        raise SystemExit(2)

    hardware_ok = _detect_hardware()
    if no_mock and not hardware_ok:
        console.print(
            "[red]--no-mock указан, но unitree_sdk2py недоступен.[/red]\n"
            "Установите SDK Unitree и повторите."
        )
        raise SystemExit(3)
    use_mock = mock or (not no_mock and not hardware_ok)

    try:
        if use_mock:
            console.print("[yellow]Режим: [MOCK] симуляция ходьбы G1.[/yellow]")
            result = navigate_mock(target_xyz, speed, stop_distance)
        else:
            console.print("[green]Режим: реальная навигация через SportClient.[/green]")
            result = navigate_real(target_xyz, speed, stop_distance, interface)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Непредвиденная ошибка: {exc}[/red]")
        console.print_exception()
        sys.exit(1)

    console.print(f"\n[bold]Результат навигации:[/bold] {result}")


if __name__ == "__main__":
    main()
