#!/usr/bin/env python3
"""grasp_controller.py — хватка стакана с контролем силы (RH56DFTP).

Стратегия:
1. **Грубое наведение** по 3D-координатам от CV (быстрое движение к точке над стаканом).
2. **Плавное сближение** на скорости ≤ 5 см/с на последних 10 см.
3. **Force-guided grasp**: пальцы медленно сходятся, контроль силы прижима
   (порог срабатывания 30–80 г, бумажный стакан деформируется при ~150 г).
4. **Anti-crush**: при силе > 200 г — мгновенный разжим.

Mock-режим: если unitree_sdk2py недоступен — скрипт симулирует движение
и силу, печатая ``[MOCK]`` в логах.

Пример::

    python grasp_controller.py --target 0.45,0.10,0.30
    python grasp_controller.py --target 0.45,0.10,0.30 --mock --calib calibration.json
"""

from __future__ import annotations

# --- Импорты ---
import json
import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table

# --- Константы ---
# Скорости движения руки (м/с)
SPEED_APPROACH = 0.20       # грубое наведение
SPEED_FINE = 0.05           # плавное сближение (5 см/с)
FINE_RANGE_M = 0.10         # последние 10 см — плавно
# Пороги силы (граммы)
FORCE_GRASP_MIN = 30.0      # минимальный порог срабатывания
FORCE_GRASP_MAX = 80.0      # максимальный рабочий порог
FORCE_DEFORM = 150.0        # порог деформации стакана
FORCE_ANTI_CRUSH = 200.0    # мгновенный разжим
# Замеры для оценки стабильности (см. stability_check.py)
DT_CONTROL_S = 0.02         # цикл управления 50 Гц

console = Console()


# --- Импорт железа ---
def _try_import_unitree():
    """Ленивый импорт unitree_sdk2py."""
    try:
        from unitree_sdk2py.core.channel.channel_factory import ChannelFactory  # type: ignore
        return ChannelFactory
    except Exception:  # noqa: BLE001
        return None


def _detect_hardware() -> bool:
    return _try_import_unitree() is not None


# --- Mock-симулятор руки/сенсоров ---
class MockArm:
    """Симулирует движение руки и измерения силы."""

    def __init__(self, calibration: Optional[dict] = None):
        self.pos = [0.0, 0.0, 0.0]  # (x, y, z) в системе базы руки
        self.finger_open = 1.0  # 0 — сжат, 1 — полностью разжат
        self.contact_force_g = 0.0
        self.calibration = calibration

    def move_towards(self, target: list[float], speed_m_s: float, dt: float) -> None:
        """Двигает руку к target с заданной скоростью."""
        for i in range(3):
            delta = target[i] - self.pos[i]
            step = speed_m_s * dt
            if abs(delta) <= step:
                self.pos[i] = target[i]
            else:
                self.pos[i] += step * (1 if delta > 0 else -1)

    def at_target(self, target: list[float], tol: float = 0.005) -> bool:
        return all(abs(self.pos[i] - target[i]) <= tol for i in range(3))

    def set_finger(self, openness: float) -> None:
        """0..1 — степень закрытия пальцев (1 = полностью открыт)."""
        self.finger_open = max(0.0, min(1.0, openness))
        # Контактная сила эмулируется: стакан «встречается» при openness ~0.55
        # и сила растёт линейно по мере закрытия
        if self.finger_open < 0.55:
            # Линейный рост силы от 0 до ~80 г при полном закрытии
            self.contact_force_g = (0.55 - self.finger_open) * 200.0
        else:
            self.contact_force_g = 0.0

    def read_force_g(self) -> float:
        # + лёгкий шум ±1 г
        noise = ((int(time.time() * 1000) % 5) - 2) * 0.5
        return max(0.0, self.contact_force_g + noise)

    def release(self) -> None:
        self.finger_open = 1.0
        self.contact_force_g = 0.0


# --- Реальная рука (каркас) ---
class RealArm:
    """Каркас реальной руки. Реализуется под конкретную версию SDK."""

    def __init__(self, interface: str, calibration: Optional[dict] = None):
        ChannelFactory = _try_import_unitree()
        if ChannelFactory is None:
            raise RuntimeError("unitree_sdk2py не установлен")
        try:
            ChannelFactory.Instance().Init(0, interface)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Не удалось инициализировать канал: {exc}") from exc
        self.calibration = calibration
        self.pos = [0.0, 0.0, 0.0]
        self.finger_open = 1.0

    def move_towards(self, target: list[float], speed_m_s: float, dt: float) -> None:
        # TODO: реальная команда через ArmClient/HighLevelApi
        for i in range(3):
            delta = target[i] - self.pos[i]
            step = speed_m_s * dt
            if abs(delta) <= step:
                self.pos[i] = target[i]
            else:
                self.pos[i] += step * (1 if delta > 0 else -1)

    def at_target(self, target: list[float], tol: float = 0.005) -> bool:
        return all(abs(self.pos[i] - target[i]) <= tol for i in range(3))

    def set_finger(self, openness: float) -> None:
        self.finger_open = max(0.0, min(1.0, openness))
        # TODO: реальная команда на моторы пальцев

    def start_wait(self) -> None:
        """No-op для совместимости с mock-интерфейсом handover.py."""
        pass

    def read_force_g(self) -> float:
        # TODO: чтение из LowState, преобразование через self.calibration
        return 0.0

    def release(self) -> None:
        self.set_finger(1.0)


# --- Контроллер хвата ---
def parse_target(target_str: str) -> list[float]:
    """Парсит 'x,y,z' → [x, y, z]."""
    try:
        parts = [float(v) for v in target_str.split(",")]
    except ValueError:
        raise click.BadParameter(f"Некорректные координаты: {target_str}")
    if len(parts) != 3:
        raise click.BadParameter("Ожидается 3 координаты через запятую: x,y,z")
    return parts


def make_status_table(arm, phase: str, force: float, target: list[float]) -> Table:
    """Таблица состояния для rich.Live."""
    table = Table(title=f"Force-guided grasp — фаза: {phase}")
    table.add_column("Параметр")
    table.add_column("Значение", justify="right")
    table.add_row("Позиция (x,y,z), м",
                  f"({arm.pos[0]:.3f}, {arm.pos[1]:.3f}, {arm.pos[2]:.3f})")
    table.add_row("Цель (x,y,z), м",
                  f"({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")
    table.add_row("Открытость пальцев", f"{arm.finger_open:.2f}")
    table.add_row("Сила, г", f"{force:.1f}")
    table.add_row("Порог хвата, г", f"{FORCE_GRASP_MIN}–{FORCE_GRASP_MAX}")
    table.add_row("Anti-crush, г", f"{FORCE_ANTI_CRUSH:.0f}")
    return table


def run_grasp(arm, target: list[float], pre_grasp_offset: float = 0.08) -> dict:
    """Полный цикл хватки. Возвращает словарь с результатом."""
    # Точка пред-хвата: над стаканом на pre_grasp_offset м
    pre_target = [target[0], target[1], target[2] + pre_grasp_offset]
    console.print(
        f"[cyan]Грубое наведение к точке над стаканом: "
        f"({pre_target[0]:.2f}, {pre_target[1]:.2f}, {pre_target[2]:.2f})[/cyan]"
    )
    arm.set_finger(1.0)  # пальцы полностью открыты

    # 1) Грубое наведение
    with Live(make_status_table(arm, "APPROACH", arm.read_force_g(), pre_target),
              refresh_per_second=20, console=console) as live:
        while not arm.at_target(pre_target):
            arm.move_towards(pre_target, SPEED_APPROACH, DT_CONTROL_S)
            force = arm.read_force_g()
            if force > FORCE_ANTI_CRUSH:
                console.print("[red]Anti-crush на этапе подхода! Разжим.[/red]")
                arm.release()
                return {"ok": False, "reason": "anti_crush_approach"}
            live.update(make_status_table(arm, "APPROACH", force, pre_target))
            time.sleep(DT_CONTROL_S)

    # 2) Плавное сближение (5 см/с) к target
    console.print("[cyan]Плавное сближение со скоростью 5 см/с ...[/cyan]")
    with Live(make_status_table(arm, "FINE_APPROACH", arm.read_force_g(), target),
              refresh_per_second=20, console=console) as live:
        timeout = time.time() + 5.0  # страховка 5 сек
        while not arm.at_target(target):
            if time.time() > timeout:
                console.print("[yellow]Таймаут плавного сближения — хватаем по текущей позиции.[/yellow]")
                break
            arm.move_towards(target, SPEED_FINE, DT_CONTROL_S)
            force = arm.read_force_g()
            if force > FORCE_ANTI_CRUSH:
                console.print("[red]Anti-crush на сближении! Разжим.[/red]")
                arm.release()
                return {"ok": False, "reason": "anti_crush_fine"}
            if force > FORCE_GRASP_MIN * 0.5:
                # Уже касаемся — переходим к хвату раньше
                break
            live.update(make_status_table(arm, "FINE_APPROACH", force, target))
            time.sleep(DT_CONTROL_S)

    # 3) Force-guided grasp: медленно закрываем пальцы
    console.print("[cyan]Force-guided grasp: схождение пальцев ...[/cyan]")
    openness = 1.0
    grasp_ok = False
    with Live(make_status_table(arm, "GRASP", arm.read_force_g(), target),
              refresh_per_second=20, console=console) as live:
        while openness > 0.0:
            openness -= 0.01  # ~20 итераций до полного закрытия
            arm.set_finger(openness)
            force = arm.read_force_g()
            live.update(make_status_table(arm, "GRASP", force, target))
            # Anti-crush — мгновенный разжим
            if force > FORCE_ANTI_CRUSH:
                console.print(f"[red]Anti-crush! Сила {force:.1f} г > {FORCE_ANTI_CRUSH} г. Разжим.[/red]")
                arm.release()
                return {"ok": False, "reason": "anti_crush_grasp", "force_g": force}
            # Превышен порог деформации — опасно
            if force > FORCE_DEFORM:
                console.print(
                    f"[red]Превышен порог деформации: {force:.1f} г > {FORCE_DEFORM} г. "
                    f"Разжим для безопасности.[/red]"
                )
                arm.release()
                return {"ok": False, "reason": "deform_exceeded", "force_g": force}
            # Попали в рабочий диапазон — хват готов
            if FORCE_GRASP_MIN <= force <= FORCE_GRASP_MAX:
                console.print(f"[green]Хват зафиксирован при силе {force:.1f} г.[/green]")
                grasp_ok = True
                break
            time.sleep(DT_CONTROL_S)

    if not grasp_ok:
        console.print("[yellow]Не удалось достичь рабочего диапазона силы. Разжим.[/yellow]")
        arm.release()
        return {"ok": False, "reason": "no_contact"}

    return {
        "ok": True,
        "final_force_g": arm.read_force_g(),
        "final_position": list(arm.pos),
        "finger_openness": arm.finger_open,
    }


@click.command()
@click.option("--target", "-t", required=True,
              help="3D-координаты стакана в системе базы руки: x,y,z (метры).")
@click.option("--mock", is_flag=True, default=False, help="Принудительно mock-режим.")
@click.option("--no-mock", is_flag=True, default=False,
              help="Принудительно реальный режим (требует unitree_sdk2py).")
@click.option("--interface", default="lo", help="Сетевой интерфейс к роботу.")
@click.option("--calib", default=None, type=click.Path(exists=True),
              help="JSON с калибровкой сенсоров (из calibrate.py).")
@click.option("--pre-grasp-offset", default=0.08, type=float,
              help="Высота точки пред-хвата над стаканом, м.")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Куда сохранить JSON с результатом.")
def main(target: str, mock: bool, no_mock: bool, interface: str, calib: Optional[str],
         pre_grasp_offset: float, output: Optional[str]) -> None:
    """Хватка стакана с контролем силы (force-guided grasp)."""
    console.print("[bold cyan]COFFEE manipulation — grasp controller[/bold cyan]")
    target_xyz = parse_target(target)
    console.print(f"Целевая позиция стакана: {target_xyz} м")

    calibration = None
    if calib:
        try:
            calibration = json.loads(Path(calib).read_text(encoding="utf-8"))
            console.print(f"[green]Калибровка загружена: {calib}[/green]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Не удалось загрузить калибровку: {exc}[/yellow]")

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
            console.print("[yellow]Режим: [MOCK] симуляция руки.[/yellow]")
            arm: MockArm | RealArm = MockArm(calibration)
        else:
            console.print("[green]Режим: реальная рука.[/green]")
            arm = RealArm(interface, calibration)
        result = run_grasp(arm, target_xyz, pre_grasp_offset=pre_grasp_offset)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Непредвиденная ошибка: {exc}[/red]")
        console.print_exception()
        sys.exit(1)

    console.print(f"\n[bold]Результат хвата:[/bold] {result}")
    if output:
        Path(output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        console.print(f"[green]Сохранено в {output}[/green]")


if __name__ == "__main__":
    main()
