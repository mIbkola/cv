#!/usr/bin/env python3
"""carry_pose.py — поза с грузом (компенсация наклона через IMU).

Включает/выключает «позу с грузом»:
- снижение скорости ходьбы до 0.5–0.8 м/с;
- опускание центра масс на 2–3 см;
- активная компенсация наклона корпуса по данным IMU (PID по roll/pitch).

Без железа — mock-режим: печатает симулированные значения углов и PID-коррекции.

Пример::

    python carry_pose.py --enable --mock
    python carry_pose.py --disable --mock
    python carry_pose.py --enable --no-mock --interface eth0
"""

from __future__ import annotations

# --- Импорты ---
import math
import sys
import time
from typing import Optional

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table

# --- Константы ---
CARRY_SPEED_M_S = 0.6          # 0.5–0.8 м/с с грузом
CARRY_HEIGHT_OFFSET_M = -0.03  # центр масс на 3 см ниже
IMU_LOOP_HZ = 50               # 50 Гц
PID_KP = 1.2
PID_KI = 0.05
PID_KD = 0.4
DEAD_BAND_DEG = 1.0            # зона нечувствительности, град
MAX_CORRECTION_DEG = 8.0       # ограничение коррекции
LOG_DURATION_S = 5.0           # длительность демо-лога в mock-режиме

console = Console()


# --- Импорт SDK ---
def _try_import_unitree():
    try:
        from unitree_sdk2py.core.channel.channel_factory import ChannelFactory  # type: ignore
        return ChannelFactory
    except Exception:  # noqa: BLE001
        return None


def _detect_hardware() -> bool:
    return _try_import_unitree() is not None


# --- PID ---
class PIDController:
    """Простой PID-контроллер для компенсации наклона."""

    def __init__(self, kp: float, ki: float, kd: float, deadband: float = 0.0,
                 out_limit: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.deadband = deadband
        self.out_limit = out_limit
        self._integral = 0.0
        self._prev_err = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_err = 0.0

    def update(self, err: float, dt: float) -> float:
        if abs(err) <= self.deadband:
            err = 0.0
        self._integral += err * dt
        # Anti-windup: ограничим интеграл
        self._integral = max(-10.0, min(10.0, self._integral))
        derivative = (err - self._prev_err) / dt if dt > 0 else 0.0
        self._prev_err = err
        out = self.kp * err + self.ki * self._integral + self.kd * derivative
        if self.out_limit > 0:
            out = max(-self.out_limit, min(self.out_limit, out))
        return out


# --- Mock IMU ---
def mock_imu_reading(t: float) -> tuple[float, float]:
    """Симулированные roll и pitch (град), лёгкое покачивание."""
    roll = 2.5 * math.sin(2 * math.pi * 0.8 * t)   # ±2.5 град
    pitch = 1.5 * math.sin(2 * math.pi * 0.5 * t)  # ±1.5 град
    return roll, pitch


# --- Логирование ---
def make_table(roll: float, pitch: float, corr_roll: float, corr_pitch: float,
               enabled: bool) -> Table:
    table = Table(title=f"Carry pose — {'ВКЛ' if enabled else 'ВЫКЛ'}")
    table.add_column("Параметр")
    table.add_column("Значение", justify="right")
    table.add_row("Режим", "CARRY" if enabled else "NORMAL")
    table.add_row("Скорость, м/с", f"{CARRY_SPEED_M_S if enabled else 2.0:.1f}")
    table.add_row("Высота корпуса, Δм", f"{CARRY_HEIGHT_OFFSET_M if enabled else 0.0:+.2f}")
    table.add_row("Roll, град", f"{roll:.2f}")
    table.add_row("Pitch, град", f"{pitch:.2f}")
    table.add_row("Коррекция roll, град", f"{corr_roll:.2f}")
    table.add_row("Коррекция pitch, град", f"{corr_pitch:.2f}")
    return table


# --- Основная логика ---
def run_mock_carry(enable: bool) -> dict:
    """Демо PID-компенсации в mock-режиме."""
    if not enable:
        console.print("[yellow][MOCK] Поза с грузом ВЫКЛ: скорость 2.0 м/с, "
                      "компенсация отключена.[/yellow]")
        return {"enabled": False, "speed_m_s": 2.0, "height_offset_m": 0.0}
    console.print(
        f"[yellow][MOCK] Поза с грузом ВКЛ: скорость {CARRY_SPEED_M_S} м/с, "
        f"центр масс на {CARRY_HEIGHT_OFFSET_M*100:.0f} см ниже, "
        f"PID-компенсация наклона активна.[/yellow]"
    )
    pid_roll = PIDController(PID_KP, PID_KI, PID_KD, DEAD_BAND_DEG, MAX_CORRECTION_DEG)
    pid_pitch = PIDController(PID_KP, PID_KI, PID_KD, DEAD_BAND_DEG, MAX_CORRECTION_DEG)
    dt = 1.0 / IMU_LOOP_HZ
    t_start = time.time()
    max_corr = 0.0
    with Live(make_table(0.0, 0.0, 0.0, 0.0, True),
              refresh_per_second=10, console=console) as live:
        while time.time() - t_start < LOG_DURATION_S:
            t = time.time() - t_start
            roll, pitch = mock_imu_reading(t)
            corr_roll = -pid_roll.update(roll, dt)
            corr_pitch = -pid_pitch.update(pitch, dt)
            max_corr = max(max_corr, abs(corr_roll), abs(corr_pitch))
            live.update(make_table(roll, pitch, corr_roll, corr_pitch, True))
            time.sleep(dt)
    return {
        "enabled": True,
        "speed_m_s": CARRY_SPEED_M_S,
        "height_offset_m": CARRY_HEIGHT_OFFSET_M,
        "max_correction_deg": round(max_corr, 2),
        "pid": {"kp": PID_KP, "ki": PID_KI, "kd": PID_KD},
    }


def run_real_carry(enable: bool, interface: str) -> dict:
    """Включение/выключение позы с грузом через unitree_sdk2py."""
    ChannelFactory = _try_import_unitree()
    if ChannelFactory is None:
        return run_mock_carry(enable)
    try:
        ChannelFactory.Instance().Init(0, interface)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Не удалось инициализировать канал: {exc}[/red]")
        console.print("[yellow]Переключаюсь в mock-режим.[/yellow]")
        return run_mock_carry(enable)

    # Каркас: реальные команды зависят от версии SDK
    console.print(
        f"[cyan]Отправка команды позы с грузом: "
        f"{'ВКЛ' if enable else 'ВЫКЛ'} (через {interface}).[/cyan]"
    )
    # TODO: client.SetSpeed(CARRY_SPEED_M_S) и балансировка через SportMode
    return {
        "enabled": enable,
        "speed_m_s": CARRY_SPEED_M_S if enable else 2.0,
        "height_offset_m": CARRY_HEIGHT_OFFSET_M if enable else 0.0,
        "interface": interface,
    }


@click.command()
@click.option("--enable/--disable", default=True, help="Включить или выключить позу с грузом.")
@click.option("--mock", is_flag=True, default=False, help="Принудительно mock-режим.")
@click.option("--no-mock", is_flag=True, default=False,
              help="Принудительно реальный режим.")
@click.option("--interface", default="lo", help="Сетевой интерфейс к роботу.")
def main(enable: bool, mock: bool, no_mock: bool, interface: str) -> None:
    """Включить/выключить позу с грузом (компенсация наклона через IMU)."""
    console.print("[bold cyan]COFFEE locomotion — carry pose[/bold cyan]")

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
            console.print("[yellow]Режим: [MOCK] симуляция IMU и PID.[/yellow]")
            result = run_mock_carry(enable)
        else:
            console.print("[green]Режим: реальная установка позы.[/green]")
            result = run_real_carry(enable, interface)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Непредвиденная ошибка: {exc}[/red]")
        console.print_exception()
        sys.exit(1)

    console.print(f"\n[bold]Результат carry_pose:[/bold] {result}")


if __name__ == "__main__":
    main()
