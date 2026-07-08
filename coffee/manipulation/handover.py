#!/usr/bin/env python3
"""handover.py — передача стакана Олегу.

После подхода к Олегу робот поднимает руку со стаканом на уровень груди
Олега (~1.1 м от пола) и ждёт, пока Олег возьмёт стакан. Признак приёма:
сила на пальцах упала ниже 20 г (пальцы больше не чувствуют стакан) →
разжать пальцы и вернуть руку в нейтральное положение.

Mock-режим: симулирует ожидание приёма и плавное падение силы.

Пример::

    python handover.py --mock --accept-after 2.0
    python handover.py --no-mock --interface eth0
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
CHEST_HEIGHT_M = 1.10      # уровень груди Олега (типичное значение)
HANDOFF_DISTANCE_M = 0.70  # робот стоит в 0.7 м от Олега
RELEASE_FORCE_G = 20.0     # сила < 20 г → стакан принят
WAIT_TIMEOUT_S = 30.0      # ждать приёма не дольше 30 сек (см. ТЗ риски)
LIFT_SPEED_M_S = 0.10      # поднятие руки — медленно
DT_CONTROL_S = 0.02

console = Console()


# --- Железо ---
def _try_import_unitree():
    try:
        from unitree_sdk2py.core.channel.channel_factory import ChannelFactory  # type: ignore
        return ChannelFactory
    except Exception:  # noqa: BLE001
        return None


def _detect_hardware() -> bool:
    return _try_import_unitree() is not None


# --- Mock-рука ---
class MockHandoverArm:
    """Симулирует руку со стаканом и ожидание приёма."""

    def __init__(self, accept_after_s: float = 2.0):
        self.pos = [0.45, 0.10, 0.90]   # рука в исходной позиции (у талии)
        self.finger_open = 0.30          # удерживает стакан
        self.force_g = 55.0
        # Таймер «ожидания приёма» стартует только когда робот готов отдать
        # стакан (после подъёма на уровень груди). См. start_wait().
        self.wait_t0: float | None = None
        self.accept_after = accept_after_s

    def move_towards(self, target: list[float], speed_m_s: float, dt: float) -> None:
        for i in range(3):
            delta = target[i] - self.pos[i]
            step = speed_m_s * dt
            if abs(delta) <= step:
                self.pos[i] = target[i]
            else:
                self.pos[i] += step * (1 if delta > 0 else -1)

    def at_target(self, target: list[float], tol: float = 0.005) -> bool:
        return all(abs(self.pos[i] - target[i]) <= tol for i in range(3))

    def start_wait(self) -> None:
        """Начать отсчёт времени ожидания приёма (mock-симуляция)."""
        self.wait_t0 = time.time()

    def read_force_g(self) -> float:
        # До старта ожидания — удерживаем стабильную силу (стакан в руке).
        if self.wait_t0 is None:
            return self.force_g
        elapsed = time.time() - self.wait_t0
        if elapsed > self.accept_after:
            # Плавный спад силы (Олег берёт стакан)
            return max(0.0, self.force_g - (elapsed - self.accept_after) * 200.0)
        return self.force_g

    def release(self) -> None:
        self.finger_open = 1.0
        self.force_g = 0.0


# --- Логика передачи ---
def make_table(phase: str, force: float, elapsed: float, pos: list[float]) -> Table:
    table = Table(title=f"Handover — фаза: {phase}")
    table.add_column("Параметр")
    table.add_column("Значение", justify="right")
    table.add_row("Фаза", phase)
    table.add_row("Позиция (x,y,z), м",
                  f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    table.add_row("Сила, г", f"{force:.2f}")
    table.add_row("Порог отпускания, г", f"{RELEASE_FORCE_G}")
    table.add_row("Прошло, с", f"{elapsed:.1f}")
    table.add_row("Таймаут, с", f"{WAIT_TIMEOUT_S}")
    return table


def run_handover(arm, target_height: float = CHEST_HEIGHT_M) -> dict:
    """Поднять руку на уровень груди и дождаться приёма."""
    # Целевая позиция: рука вытянута вперёд на HANDOFF_DISTANCE_M, высота груди
    target = [HANDOFF_DISTANCE_M, 0.0, target_height]
    console.print(
        f"[cyan]Поднятие руки на уровень груди Олега "
        f"(z={target_height:.2f} м), вытянута на {HANDOFF_DISTANCE_M:.2f} м ...[/cyan]"
    )

    # Фаза 1: подъём руки
    with Live(make_table("LIFT", arm.read_force_g(), 0.0, arm.pos),
              refresh_per_second=20, console=console) as live:
        while not arm.at_target(target):
            arm.move_towards(target, LIFT_SPEED_M_S, DT_CONTROL_S)
            force = arm.read_force_g()
            # Контроль: стакан не выскользнул во время подъёма
            if force < 5.0:
                console.print(
                    "[red]Стакан выскользнул при подъёме руки! "
                    "Возврат на базу.[/red]"
                )
                return {"ok": False, "reason": "lost_grip"}
            live.update(make_table("LIFT", force, 0.0, arm.pos))
            time.sleep(DT_CONTROL_S)

    # Фаза 2: ожидание приёма (сила < RELEASE_FORCE_G)
    console.print(
        f"[cyan]Ожидаю приём стакана (сила < {RELEASE_FORCE_G} г) "
        f"до {WAIT_TIMEOUT_S:.0f} сек ...[/cyan]"
    )
    # Стартуем «симуляцию приёма» только сейчас (mock) — рука уже в позиции
    if hasattr(arm, "start_wait"):
        arm.start_wait()
    t_start = time.time()
    accepted = False
    with Live(make_table("WAIT", arm.read_force_g(), 0.0, arm.pos),
              refresh_per_second=20, console=console) as live:
        while time.time() - t_start < WAIT_TIMEOUT_S:
            force = arm.read_force_g()
            elapsed = time.time() - t_start
            live.update(make_table("WAIT", force, elapsed, arm.pos))
            if force < RELEASE_FORCE_G:
                console.print(
                    f"[green]Стакан принят! Сила {force:.1f} г < {RELEASE_FORCE_G} г. "
                    f"Отпускаю пальцы.[/green]"
                )
                accepted = True
                break
            time.sleep(DT_CONTROL_S)

    if not accepted:
        # По ТЗ риски: Олега нет → вернуть стакан на стол → сообщить голосом
        console.print(
            f"[red]Таймаут {WAIT_TIMEOUT_S:.0f} сек — Олег не принял стакан. "
            f"Возврат стакана на базу + голосовое сообщение.[/red]"
        )
        # Возврат: опустить руку обратно
        return_target = [0.30, 0.10, 0.70]
        with Live(make_table("RETURN", arm.read_force_g(), 0.0, arm.pos),
                  refresh_per_second=20, console=console) as live:
            while not arm.at_target(return_target):
                arm.move_towards(return_target, LIFT_SPEED_M_S, DT_CONTROL_S)
                force = arm.read_force_g()
                live.update(make_table("RETURN", force, 0.0, arm.pos))
                time.sleep(DT_CONTROL_S)
        return {"ok": False, "reason": "timeout", "force_final_g": arm.read_force_g()}

    # Фаза 3: разжим пальцев
    console.print("[cyan]Разжим пальцев ...[/cyan]")
    arm.release()
    time.sleep(0.5)
    return {
        "ok": True,
        "reason": "accepted",
        "force_final_g": arm.read_force_g(),
        "wait_s": time.time() - t_start,
    }


@click.command()
@click.option("--mock", is_flag=True, default=False, help="Принудительно mock-режим.")
@click.option("--no-mock", is_flag=True, default=False,
              help="Принудительно реальный режим.")
@click.option("--interface", default="lo", help="Сетевой интерфейс к роботу.")
@click.option("--chest-height", default=CHEST_HEIGHT_M, type=float,
              help="Высота груди Олега от пола, м.")
@click.option("--accept-after", default=2.0, type=float,
              help="В mock-режиме: через сколько секунд симулировать приём.")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Куда сохранить JSON-результат.")
def main(mock: bool, no_mock: bool, interface: str, chest_height: float,
         accept_after: float, output: Optional[str]) -> None:
    """Передача стакана Олегу: поднять на уровень груди, дождаться приёма."""
    console.print("[bold cyan]COFFEE manipulation — handover[/bold cyan]")

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
            console.print(f"[yellow]Режим: [MOCK] симуляция руки "
                          f"(accept_after={accept_after}s).[/yellow]")
            arm = MockHandoverArm(accept_after_s=accept_after)
        else:
            console.print("[green]Режим: реальная рука.[/green]")
            from grasp_controller import RealArm  # type: ignore
            arm = RealArm(interface)
        result = run_handover(arm, target_height=chest_height)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Непредвиденная ошибка: {exc}[/red]")
        console.print_exception()
        sys.exit(1)

    console.print(f"\n[bold]Результат handover:[/bold] {result}")
    if output:
        Path(output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        console.print(f"[green]Сохранено в {output}[/green]")


if __name__ == "__main__":
    main()
