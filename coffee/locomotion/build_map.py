#!/usr/bin/env python3
"""build_map.py — построение карты через Livox MID-360.

Без железа работает в mock-режиме: генерирует «карту» комнаты (плоский пол
+ несколько препятствий) в виде облака точек и сохраняет в ``.pcd`` (текстовый
формат) или ``.npz``. Если установлен ``open3d`` — используется для записи.

Пример::

    python build_map.py --mock --output map.pcd
    python build_map.py --no-mock --scan-time 30 --output map.pcd
"""

from __future__ import annotations

# --- Импорты ---
import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table

# --- Константы ---
ROOM_SIZE_M = 5.0          # размер комнаты 5×5 м (для mock)
N_OBSTACLES = 4            # число препятствий в mock-карте
POINTS_PER_OBSTACLE = 200
N_FLOOR_POINTS = 1500
DEFAULT_SCAN_TIME_S = 30

console = Console()


# --- Импорт SDK ---
def _try_import_unitree():
    """Ленивый импорт unitree_sdk2py (для проверки наличия SDK)."""
    try:
        from unitree_sdk2py.core.channel.channel_factory import ChannelFactory  # type: ignore
        return ChannelFactory
    except Exception:  # noqa: BLE001
        return None


def _detect_livox() -> bool:
    """True если установлен livox_ros_driver2 / livox-sdk2."""
    try:
        import livox_sdk2  # type: ignore  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        pass
    try:
        import livox_ros_driver2  # type: ignore  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _detect_hardware() -> bool:
    return _try_import_unitree() is not None


def _try_import_open3d():
    try:
        import open3d as o3d  # type: ignore
        return o3d
    except ImportError:
        return None


# --- Mock-карта ---
def generate_mock_map() -> "list[list[float]]":
    """Генерирует простую mock-карту: пол + несколько кубоидов."""
    import random
    rng = random.Random(42)
    points: list[list[float]] = []

    # Пол — точки на z=0
    for _ in range(N_FLOOR_POINTS):
        x = rng.uniform(-ROOM_SIZE_M / 2, ROOM_SIZE_M / 2)
        y = rng.uniform(-ROOM_SIZE_M / 2, ROOM_SIZE_M / 2)
        points.append([x, y, 0.0])

    # Препятствия — кубоиды 0.5×0.5×0.8 м
    for i in range(N_OBSTACLES):
        cx = rng.uniform(-2.0, 2.0)
        cy = rng.uniform(-2.0, 2.0)
        for _ in range(POINTS_PER_OBSTACLE):
            x = cx + rng.uniform(-0.25, 0.25)
            y = cy + rng.uniform(-0.25, 0.25)
            z = rng.uniform(0.0, 0.8)
            points.append([x, y, z])

    # Стены — точки по периметру
    for _ in range(800):
        side = rng.randint(0, 3)
        if side == 0:
            x = rng.uniform(-ROOM_SIZE_M / 2, ROOM_SIZE_M / 2)
            y = -ROOM_SIZE_M / 2
        elif side == 1:
            x = rng.uniform(-ROOM_SIZE_M / 2, ROOM_SIZE_M / 2)
            y = ROOM_SIZE_M / 2
        elif side == 2:
            x = -ROOM_SIZE_M / 2
            y = rng.uniform(-ROOM_SIZE_M / 2, ROOM_SIZE_M / 2)
        else:
            x = ROOM_SIZE_M / 2
            y = rng.uniform(-ROOM_SIZE_M / 2, ROOM_SIZE_M / 2)
        z = rng.uniform(0.0, 1.5)
        points.append([x, y, z])

    return points


# --- Сохранение ---
def save_pcd(points: list[list[float]], path: Path) -> None:
    """Сохраняет облако точек в простом ASCII PCD."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")
        for p in points:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")


def save_open3d(points: list[list[float]], path: Path) -> None:
    """Сохраняет через open3d (если установлен)."""
    o3d = _try_import_open3d()
    if o3d is None:
        save_pcd(points, path)
        return
    import numpy as np
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype="float64"))
    o3d.io.write_point_cloud(str(path), pcd)


# --- Реальное сканирование ---
def scan_real_livox(scan_time_s: int, interface: str) -> list[list[float]]:
    """Сбор точек с Livox MID-360 через livox_sdk2 (каркас)."""
    try:
        import livox_sdk2 as lsdk  # type: ignore
    except ImportError:
        console.print(
            "[red]livox_sdk2 не установлен. Невозможно выполнить реальное сканирование.[/red]\n"
            "Установите Livox-SDK2 и привязки Python."
        )
        raise SystemExit(3)

    console.print(f"[cyan]Подключение к Livox MID-360 через {interface} ...[/cyan]")
    # Реальная инициализация Livox-SDK2 здесь опускается — зависит от версии.
    # Заглушка: вернём mock-карту, но пометим режим как fallback.
    console.print(
        "[yellow]Внимание: реальный драйвер Livox не подключён — возвращаю mock-карту.[/yellow]"
    )
    return generate_mock_map()


def make_progress_table(elapsed: float, total: float, n_points: int) -> Table:
    table = Table(title="Сканирование карты")
    table.add_column("Параметр")
    table.add_column("Значение", justify="right")
    table.add_row("Прошло, с", f"{elapsed:.1f}")
    table.add_row("Всего, с", f"{total:.1f}")
    table.add_row("Точек собрано", str(n_points))
    return table


@click.command()
@click.option("--mock", is_flag=True, default=False, help="Принудительно mock-режим.")
@click.option("--no-mock", is_flag=True, default=False,
              help="Принудительно реальный режим (требует Livox SDK2).")
@click.option("--scan-time", default=DEFAULT_SCAN_TIME_S, type=int,
              help="Длительность сканирования, сек.")
@click.option("--interface", default="lo", help="Сетевой интерфейс.")
@click.option("--output", "-o", default="map.pcd", type=click.Path(),
              help="Куда сохранить карту (.pcd).")
def main(mock: bool, no_mock: bool, scan_time: int, interface: str,
         output: str) -> None:
    """Построение карты окружения через Livox MID-360."""
    console.print("[bold cyan]COFFEE locomotion — build map[/bold cyan]")

    if mock and no_mock:
        console.print("[red]--mock и --no-mock взаимоисключающие.[/red]")
        raise SystemExit(2)

    hardware_ok = _detect_livox() and _detect_hardware()
    if no_mock and not hardware_ok:
        console.print(
            "[red]--no-mock указан, но Livox-SDK2 / unitree_sdk2py недоступны.[/red]"
        )
        raise SystemExit(3)
    use_mock = mock or (not no_mock and not hardware_ok)

    try:
        if use_mock:
            console.print(f"[yellow]Режим: [MOCK] генерация карты комнаты "
                          f"{ROOM_SIZE_M}×{ROOM_SIZE_M} м.[/yellow]")
            # Симулируем прогресс-бар
            n_points = 0
            with Live(make_progress_table(0.0, float(scan_time), n_points),
                      refresh_per_second=10, console=console) as live:
                for i in range(scan_time):
                    time.sleep(1.0)
                    n_points += 250  # ~250 точек/сек
                    live.update(make_progress_table(float(i + 1), float(scan_time), n_points))
            points = generate_mock_map()
            console.print(f"[green]Сгенерировано {len(points)} точек.[/green]")
        else:
            console.print("[green]Режим: реальное сканирование Livox MID-360.[/green]")
            n_points = 0
            with Live(make_progress_table(0.0, float(scan_time), n_points),
                      refresh_per_second=10, console=console) as live:
                start = time.time()
                while time.time() - start < scan_time:
                    time.sleep(1.0)
                    n_points += 10000  # условно
                    live.update(make_progress_table(time.time() - start,
                                                   float(scan_time), n_points))
            points = scan_real_livox(scan_time, interface)
        save_open3d(points, Path(output))
        console.print(f"[green]Карта сохранена в {output} ({len(points)} точек).[/green]")
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Непредвиденная ошибка: {exc}[/red]")
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
