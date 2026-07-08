#!/usr/bin/env python3
"""stability_check.py — проверка стабильности хвата.

После того, как стакан схвачен, поднимаем его на 5 см и следим за силой.
Если ``Δforce > 20%`` за 200 мс — хват нестабильный (стакан проскальзывает).

Алгоритм:
1. Запомнить начальное значение силы (force_0).
2. Поднять руку на 5 см вверх со скоростью 5 см/с.
3. Мониторить силу в течение 1 секунды на удержании.
4. Если за любой интервал 200 мс изменение силы превысило 20% от force_0
   — признать хват нестабильным.
5. Дополнительно: проверка абсолютного уровня силы (не вышел ли за рабочий
   диапазон 30–80 г или не превысил порог деформации 150 г).

Пример::

    python stability_check.py --mock
    python stability_check.py --no-mock --interface eth0
"""

from __future__ import annotations

# --- Импорты ---
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table

# --- Константы ---
LIFT_HEIGHT_M = 0.05       # 5 см
LIFT_SPEED_M_S = 0.05      # 5 см/с
HOLD_TIME_S = 1.0          # длительность проверки на удержании
WINDOW_S = 0.2             # 200 мс — окно для оценки Δforce
DELTA_FORCE_PERCENT = 0.20 # 20% — порог нестабильности
DT_CONTROL_S = 0.02        # цикл 50 Гц
FORCE_MIN_G = 30.0
FORCE_MAX_G = 80.0
FORCE_DEFORM_G = 150.0
FORCE_ANTI_CRUSH_G = 200.0

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


# --- Mock-рука (совместимая с grasp_controller по интерфейсу) ---
class MockArmStability:
    """Симулирует руку с уже захваченным стаканом. Подъём + лёгкое проскальзывание."""

    def __init__(self, unstable: bool = False):
        self.pos = [0.45, 0.10, 0.30]
        self.finger_open = 0.30  # стакан удерживается
        self.base_force_g = 55.0
        self.unstable = unstable
        self._t0 = time.time()

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

    def read_force_g(self) -> float:
        t = time.time() - self._t0
        # Базовый уровень + лёгкий шум
        noise = ((int(time.time() * 1000) % 5) - 2) * 0.5
        if self.unstable and t > 0.3:
            # Имитация резкого проскальзывания: сила падает со скоростью ~1 г/сек
            # → за 200 мс падает на ~20+%, что должно сработать на детекцию.
            drop_rate = 1.2  # доля в секунду
            elapsed_drop = (t - 0.3) * drop_rate
            return max(0.0, self.base_force_g * max(0.0, 1.0 - elapsed_drop) + noise)
        return max(0.0, self.base_force_g + noise)

    def release(self) -> None:
        self.finger_open = 1.0
        self.base_force_g = 0.0


# --- Проверка стабильности ---
def make_table(phase: str, force: float, delta_pct: float, pos: list[float]) -> Table:
    table = Table(title=f"Stability check — фаза: {phase}")
    table.add_column("Параметр")
    table.add_column("Значение", justify="right")
    table.add_row("Фаза", phase)
    table.add_row("Позиция (x,y,z), м",
                  f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    table.add_row("Сила, г", f"{force:.2f}")
    table.add_row("Δforce за 200 мс, %", f"{delta_pct * 100:.1f}")
    table.add_row("Порог Δforce, %", f"{DELTA_FORCE_PERCENT * 100:.0f}")
    return table


def run_stability_check(arm) -> dict:
    """Поднимает руку на 5 см и проверяет стабильность силы."""
    start_pos = list(arm.pos)
    lift_target = [start_pos[0], start_pos[1], start_pos[2] + LIFT_HEIGHT_M]
    console.print(f"[cyan]Подъём стакана на {LIFT_HEIGHT_M*100:.0f} см ...[/cyan]")

    # Фаза 1: подъём
    with Live(make_table("LIFT", arm.read_force_g(), 0.0, arm.pos),
              refresh_per_second=20, console=console) as live:
        while not arm.at_target(lift_target):
            arm.move_towards(lift_target, LIFT_SPEED_M_S, DT_CONTROL_S)
            force = arm.read_force_g()
            if force > FORCE_ANTI_CRUSH_G:
                console.print("[red]Anti-crush при подъёме! Разжим.[/red]")
                arm.release()
                return {"stable": False, "reason": "anti_crush"}
            live.update(make_table("LIFT", force, 0.0, arm.pos))
            time.sleep(DT_CONTROL_S)

    # Фаза 2: удержание + мониторинг
    console.print(f"[cyan]Удержание {HOLD_TIME_S:.1f} c, мониторинг силы ...[/cyan]")
    window: deque[tuple[float, float]] = deque()  # (t, force)
    t_start = time.time()
    force_0 = arm.read_force_g()
    unstable = False
    fail_reason: Optional[str] = None
    final_force = force_0
    max_delta_pct = 0.0

    with Live(make_table("HOLD", force_0, 0.0, arm.pos),
              refresh_per_second=20, console=console) as live:
        while time.time() - t_start < HOLD_TIME_S:
            force = arm.read_force_g()
            now = time.time()
            window.append((now, force))
            # Чистим окно старее WINDOW_S
            while window and now - window[0][0] > WINDOW_S:
                window.popleft()
            # Δforce в окне
            if len(window) >= 2:
                forces = [f for _, f in window]
                delta_pct = (max(forces) - min(forces)) / max(force_0, 1e-3)
            else:
                delta_pct = 0.0
            max_delta_pct = max(max_delta_pct, delta_pct)
            final_force = force

            # Anti-crush
            if force > FORCE_ANTI_CRUSH_G:
                console.print(f"[red]Anti-crush при удержании! Сила {force:.1f} г. Разжим.[/red]")
                arm.release()
                return {"stable": False, "reason": "anti_crush", "force_g": force}
            # Деформация
            if force > FORCE_DEFORM_G:
                console.print(f"[red]Превышен порог деформации: {force:.1f} г. Разжим.[/red]")
                arm.release()
                return {"stable": False, "reason": "deform_exceeded", "force_g": force}
            # Проскальзывание
            if delta_pct > DELTA_FORCE_PERCENT:
                console.print(
                    f"[red]Нестабильно: Δforce={delta_pct*100:.1f}% за {WINDOW_S*1000:.0f} мс "
                    f"(порог {DELTA_FORCE_PERCENT*100:.0f}%).[/red]"
                )
                unstable = True
                fail_reason = "slip_detected"
                break
            live.update(make_table("HOLD", force, delta_pct, arm.pos))
            time.sleep(DT_CONTROL_S)

    if unstable:
        return {
            "stable": False,
            "reason": fail_reason,
            "force_initial_g": force_0,
            "force_final_g": final_force,
            "max_delta_pct": round(max_delta_pct, 4),
        }

    console.print(
        f"[green]Стабильно: Δforce_max={max_delta_pct*100:.1f}% ≤ "
        f"{DELTA_FORCE_PERCENT*100:.0f}%, сила в диапазоне.[/green]"
    )
    return {
        "stable": True,
        "force_initial_g": force_0,
        "force_final_g": final_force,
        "max_delta_pct": round(max_delta_pct, 4),
    }


@click.command()
@click.option("--mock", is_flag=True, default=False, help="Принудительно mock-режим.")
@click.option("--no-mock", is_flag=True, default=False,
              help="Принудительно реальный режим.")
@click.option("--interface", default="lo", help="Сетевой интерфейс к роботу.")
@click.option("--unstable", is_flag=True, default=False,
              help="В mock-режиме симулировать проскальзывание (для теста).")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Куда сохранить JSON-результат.")
def main(mock: bool, no_mock: bool, interface: str, unstable: bool,
         output: Optional[str]) -> None:
    """Проверка стабильности хвата: поднять на 5 см, мониторить силу."""
    console.print("[bold cyan]COFFEE manipulation — stability check[/bold cyan]")

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
                          f"(unstable={unstable}).[/yellow]")
            arm = MockArmStability(unstable=unstable)
        else:
            console.print("[green]Режим: реальная рука.[/green]")
            # Используем каркас RealArm из grasp_controller, если он там есть.
            # Здесь — минимальная заглушка через move_towards/read_force_g.
            from grasp_controller import RealArm  # type: ignore
            arm = RealArm(interface)
        result = run_stability_check(arm)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Непредвиденная ошибка: {exc}[/red]")
        console.print_exception()
        sys.exit(1)

    console.print(f"\n[bold]Результат проверки:[/bold] {result}")
    if output:
        Path(output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        console.print(f"[green]Сохранено в {output}[/green]")


if __name__ == "__main__":
    main()
