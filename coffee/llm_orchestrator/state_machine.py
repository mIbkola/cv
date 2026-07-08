#!/usr/bin/env python3
"""state_machine.py — детерминированный конечный автомат для задачи COFFEE.

Состояния (по ТЗ)::

    IDLE → PARSE_CMD → FIND_CUP → APPROACH_CUP → GRASP
        → STABILITY_CHECK → FIND_TARGET → APPROACH_TARGET
        → HANDOVER → RELEASE → IDLE

    Переходы при ошибках:
    - STABILITY_CHECK fail (×3) → FAILED → IDLE (с голосовым сообщением)
    - FIND_CUP not_found (timeout) → FAILED → IDLE
    - HANDOVER timeout → FAILED → IDLE (стакан возвращён на базу)

Использует библиотеку ``transitions``. Если она не установлена — печатается
понятная ошибка с подсказкой установки.

Пример::

    python state_machine.py --command "принеси кофе Олегу"
    python state_machine.py --command "принеси кофе Олегу" --no-mock
"""

from __future__ import annotations

# --- Импорты ---
import json
import sys
import time
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

# --- Константы ---
STATES = [
    "IDLE",
    "PARSE_CMD",
    "FIND_CUP",
    "APPROACH_CUP",
    "GRASP",
    "STABILITY_CHECK",
    "FIND_TARGET",
    "APPROACH_TARGET",
    "HANDOVER",
    "RELEASE",
    "FAILED",
]
INITIAL_STATE = "IDLE"
MAX_STABILITY_RETRIES = 3
DEFAULT_TARGET_PERSON = "Oleg"

console = Console()


# --- Импорт transitions ---
def _import_transitions():
    try:
        from transitions import Machine  # type: ignore
        return Machine
    except ImportError as exc:  # pragma: no cover
        console.print(
            "[red]Ошибка: библиотека transitions не установлена.[/red]\n"
            "Установите: [cyan]pip install transitions[/cyan]"
        )
        raise SystemExit(2) from exc


# --- Парсер команды ---
def parse_command(command: str) -> dict:
    """Простая эвристика разбора команды.

    Возвращает {action, item, target_person}. В реальной системе этот шаг
    выполняет LLM (см. agent.py).
    """
    cmd = command.lower()
    item = "coffee"
    if "кофе" in cmd or "coffee" in cmd:
        item = "coffee"
    target_person = DEFAULT_TARGET_PERSON
    if "олег" in cmd:
        target_person = "Oleg"
    elif "маша" in cmd:
        target_person = "Masha"
    return {
        "action": "deliver_item",
        "item": item,
        "target_person": target_person,
        "raw": command,
    }


# --- Машина состояний ---
class CoffeeStateMachine:
    """Конечный автомат, реализующий флоу «принеси кофе Олегу».

    Все переходы реализованы как методы on_<transition>. Внутри они вызывают
    инструменты из tools.py. Логи пишутся через rich.
    """

    def __init__(self, mock: bool = True):
        Machine = _import_transitions()
        self.mock = mock
        self.context: dict = {}
        self.stability_retries = 0
        self.log: list[str] = []
        # Инициализация машины
        self.machine = Machine(model=self, states=STATES, initial=INITIAL_STATE,
                               send_event=False)

        # --- Переходы ---
        # IDLE → PARSE_CMD (по команде)
        self.machine.add_transition(trigger="start", source="IDLE", dest="PARSE_CMD")
        # PARSE_CMD → FIND_CUP
        self.machine.add_transition("parse_ok", "PARSE_CMD", "FIND_CUP")
        # FIND_CUP → APPROACH_CUP  /  → FAILED (не нашли)
        self.machine.add_transition("cup_found", "FIND_CUP", "APPROACH_CUP")
        self.machine.add_transition("cup_not_found", "FIND_CUP", "FAILED")
        # APPROACH_CUP → GRASP
        self.machine.add_transition("arrived_at_cup", "APPROACH_CUP", "GRASP")
        # GRASP → STABILITY_CHECK / → FAILED
        self.machine.add_transition("grasp_ok", "GRASP", "STABILITY_CHECK")
        self.machine.add_transition("grasp_fail", "GRASP", "FAILED")
        # STABILITY_CHECK → FIND_TARGET / → GRASP (retry) / → FAILED
        self.machine.add_transition("stable", "STABILITY_CHECK", "FIND_TARGET")
        self.machine.add_transition("unstable_retry", "STABILITY_CHECK", "GRASP")
        self.machine.add_transition("stable_failed", "STABILITY_CHECK", "FAILED")
        # FIND_TARGET → APPROACH_TARGET / → FAILED
        self.machine.add_transition("target_found", "FIND_TARGET", "APPROACH_TARGET")
        self.machine.add_transition("target_not_found", "FIND_TARGET", "FAILED")
        # APPROACH_TARGET → HANDOVER
        self.machine.add_transition("arrived_at_target", "APPROACH_TARGET", "HANDOVER")
        # HANDOVER → RELEASE / → FAILED (timeout)
        self.machine.add_transition("accepted", "HANDOVER", "RELEASE")
        self.machine.add_transition("handover_timeout", "HANDOVER", "FAILED")
        # RELEASE → IDLE
        self.machine.add_transition("done", "RELEASE", "IDLE")
        # FAILED → IDLE (после сообщения)
        self.machine.add_transition("reset", "FAILED", "IDLE")

    # --- Хелперы логирования ---
    def _log(self, msg: str) -> None:
        prefix = "[MOCK] " if self.mock else ""
        line = f"{prefix}{msg}"
        console.print(line)
        self.log.append(line)

    # --- Действия на переходах (вызываются вручную из run()) ---
    def do_parse_cmd(self, command: str) -> bool:
        """PARSE_CMD: разобрать команду на action/item/target_person."""
        self._log(f"PARSE_CMD: разбор команды «{command}»")
        ctx = parse_command(command)
        self.context.update(ctx)
        self._log(f"  → action={ctx['action']}, item={ctx['item']}, "
                  f"target_person={ctx['target_person']}")
        return True

    def do_find_cup(self) -> Optional[dict]:
        """FIND_CUP: найти стакан через CV."""
        from tools import find_object  # type: ignore
        self._log("FIND_CUP: поиск стакана через CV ...")
        res = find_object("cup")
        if res.get("found"):
            self.context["cup_pose"] = res.get("xyz_camera")
            self._log(f"  → стакан найден: {res.get('xyz_camera')} "
                      f"(conf={res.get('confidence')})")
            return res
        self._log("  → стакан НЕ найден.")
        return None

    def do_approach_cup(self) -> bool:
        """APPROACH_CUP: доехать к стакану (остановка в 0.5 м)."""
        from tools import navigate_to  # type: ignore
        pose = self.context.get("cup_pose")
        if not pose:
            self._log("APPROACH_CUP: нет координат стакана — abort.")
            return False
        self._log(f"APPROACH_CUP: еду к стакану {pose[:2]} ...")
        res = navigate_to(pose[0], pose[1], speed=0.6)
        ok = bool(res.get("ok"))
        if ok:
            self._log(f"  → рядом со стаканом (dist={res.get('distance_to_target')}).")
        return ok

    def do_grasp(self) -> bool:
        """GRASP: взять стакан с контролем силы."""
        from tools import grasp_with_force_feedback  # type: ignore
        pose = self.context.get("cup_pose")
        if not pose:
            self._log("GRASP: нет координат стакана — abort.")
            return False
        self._log(f"GRASP: хватка стакана {pose} ...")
        res = grasp_with_force_feedback(pose)
        ok = bool(res.get("ok"))
        if ok:
            self._log(f"  → стакан взят, сила={res.get('final_force_g')} г.")
        else:
            self._log(f"  → хватка не удалась: {res.get('reason')}")
        return ok

    def do_stability_check(self) -> bool:
        """STABILITY_CHECK: проверить стабильность."""
        from tools import verify_grasp  # type: ignore
        self._log("STABILITY_CHECK: проверка стабильности хвата ...")
        res = verify_grasp()
        if res.get("stable"):
            self._log(f"  → стабильно (Δforce={res.get('max_delta_pct')}).")
            return True
        self.stability_retries += 1
        self._log(f"  → нестабильно (попытка {self.stability_retries}/"
                  f"{MAX_STABILITY_RETRIES}).")
        return False

    def do_find_target(self) -> Optional[dict]:
        """FIND_TARGET: найти Олега по лицу."""
        from tools import find_person  # type: ignore
        name = self.context.get("target_person", DEFAULT_TARGET_PERSON)
        self._log(f"FIND_TARGET: поиск {name} по лицу ...")
        res = find_person(name)
        if res.get("found"):
            self.context["target_pose"] = res.get("xyz_camera")
            self._log(f"  → {name} найден: {res.get('xyz_camera')}")
            return res
        self._log(f"  → {name} НЕ найден.")
        return None

    def do_approach_target(self) -> bool:
        """APPROACH_TARGET: подойти к Олегу."""
        from tools import navigate_to  # type: ignore
        pose = self.context.get("target_pose")
        if not pose:
            self._log("APPROACH_TARGET: нет координат Олега — abort.")
            return False
        self._log(f"APPROACH_TARGET: еду к {self.context.get('target_person')} {pose[:2]} ...")
        res = navigate_to(pose[0], pose[1], speed=0.6)
        return bool(res.get("ok"))

    def do_handover(self) -> bool:
        """HANDOVER: передать стакан."""
        from tools import handover as tool_handover  # type: ignore
        self._log("HANDOVER: передача стакана ...")
        res = tool_handover()
        return bool(res.get("ok"))

    def do_release(self) -> None:
        """RELEASE: отпустить стакан (пальцы разжаты в handover, но явно)."""
        from tools import speak  # type: ignore
        self._log("RELEASE: стакан отпущен.")
        speak(f"Пожалуйста, {self.context.get('target_person', '')}.")

    def do_failed_announce(self, reason: str) -> None:
        """FAILED: сообщить голосом о неудаче и сбросить контекст."""
        from tools import speak  # type: ignore
        self._log(f"FAILED: {reason}")
        speak(f"Не смог выполнить задачу: {reason}.")

    # --- Главный цикл ---
    def run(self, command: str) -> dict:
        """Прогнать автомат по команде. Возвращает итоговый отчёт."""
        t_start = time.time()
        self._log(f"=== Старт задачи: «{command}» ===")

        # IDLE → PARSE_CMD
        self.start()
        if not self.do_parse_cmd(command):
            self.do_failed_announce("parse_error")
            self.reset()
            return self._report(time.time() - t_start, ok=False, reason="parse_error")
        self.parse_ok()

        # FIND_CUP
        if not self.do_find_cup():
            self.cup_not_found()
            self.do_failed_announce("cup_not_found")
            self.reset()
            return self._report(time.time() - t_start, ok=False, reason="cup_not_found")
        self.cup_found()

        # APPROACH_CUP
        if not self.do_approach_cup():
            self.do_failed_announce("approach_cup_failed")
            self.reset()
            return self._report(time.time() - t_start, ok=False, reason="approach_cup_failed")
        self.arrived_at_cup()

        # GRASP (+ STABILITY_CHECK с retry)
        while True:
            if not self.do_grasp():
                self.grasp_fail()
                self.do_failed_announce("grasp_failed")
                self.reset()
                return self._report(time.time() - t_start, ok=False, reason="grasp_failed")
            self.grasp_ok()

            # STABILITY_CHECK
            if self.do_stability_check():
                self.stable()
                break
            if self.stability_retries >= MAX_STABILITY_RETRIES:
                self.stable_failed()
                self.do_failed_announce("stability_failed")
                self.reset()
                return self._report(time.time() - t_start, ok=False,
                                    reason="stability_failed")
            self.unstable_retry()

        # FIND_TARGET
        if not self.do_find_target():
            self.target_not_found()
            self.do_failed_announce("target_not_found")
            self.reset()
            return self._report(time.time() - t_start, ok=False, reason="target_not_found")
        self.target_found()

        # APPROACH_TARGET
        if not self.do_approach_target():
            self.do_failed_announce("approach_target_failed")
            self.reset()
            return self._report(time.time() - t_start, ok=False, reason="approach_target_failed")
        self.arrived_at_target()

        # HANDOVER
        if not self.do_handover():
            self.handover_timeout()
            self.do_failed_announce("handover_timeout")
            self.reset()
            return self._report(time.time() - t_start, ok=False, reason="handover_timeout")
        self.accepted()

        # RELEASE
        self.do_release()
        self.done()
        elapsed = time.time() - t_start
        self._log(f"=== Задача завершена за {elapsed:.1f} сек ===")
        return self._report(elapsed, ok=True, reason="ok")

    def _report(self, elapsed: float, ok: bool, reason: str) -> dict:
        return {
            "ok": ok,
            "reason": reason,
            "final_state": self.state,
            "elapsed_s": round(elapsed, 2),
            "context": self.context,
            "stability_retries": self.stability_retries,
            "log": self.log,
        }


@click.command()
@click.option("--command", "-c", required=True, help="Команда, например «принеси кофе Олегу».")
@click.option("--mock/--no-mock", default=True, help="mock-режим (без железа).")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Куда сохранить JSON-отчёт.")
def main(command: str, mock: bool, output: str | None) -> None:
    """Прогнать конечный автомат по заданной команде."""
    console.print("[bold cyan]COFFEE orchestrator — state machine[/bold cyan]")
    if mock:
        console.print("[yellow]Режим: [MOCK] (без железа).[/yellow]")
    try:
        sm = CoffeeStateMachine(mock=mock)
        result = sm.run(command)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Непредвиденная ошибка: {exc}[/red]")
        console.print_exception()
        sys.exit(1)

    # Сводная таблица
    table = Table(title="Итог выполнения задачи")
    table.add_column("Параметр")
    table.add_column("Значение", justify="right")
    table.add_row("Успех", "✅" if result["ok"] else "❌")
    table.add_row("Причина", result["reason"])
    table.add_row("Финальное состояние", result["final_state"])
    table.add_row("Затрачено, сек", f"{result['elapsed_s']:.2f}")
    table.add_row("Попыток стабилизации", str(result["stability_retries"]))
    console.print(table)

    if output:
        from pathlib import Path
        Path(output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        console.print(f"[green]Отчёт сохранён в {output}[/green]")


if __name__ == "__main__":
    main()
