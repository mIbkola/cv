#!/usr/bin/env python3
"""agent.py — LLM-агент через vLLM (Qwen2.5-7B-Instruct) с function calling.

Гибридная схема:
1. LLM разбирает интент команды (action, item, target_person).
2. Дальше — детерминированная стейт-машина (state_machine.py) выполняет шаги,
   но на каждом переходе может вызывать LLM для «размышления» (опционально,
   ``--llm-steer``).
3. Если vLLM недоступен — fallback на state_machine с эвристическим парсером.

Endpoint по умолчанию: http://localhost:8000/v1/chat/completions
Модель по умолчанию: Qwen/Qwen2.5-7B-Instruct

Пример::

    python agent.py --command "принеси кофе Олегу"
    python agent.py --command "принеси кофе Олегу" --llm qwen-local --steer
"""

from __future__ import annotations

# --- Импорты ---
import json
import sys
from pathlib import Path
from typing import Any, Optional

import click
import urllib.error
import urllib.request
from rich.console import Console
from rich.table import Table

# --- Константы ---
DEFAULT_LLM_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1024
LLM_TIMEOUT_S = 30.0

SYSTEM_PROMPT = (
    "Ты — оркестратор робота Unitree G1. Получаешь команду на русском языке, "
    "разбираешь её на intent и параметры (action, item, target_person). "
    "Если команда понятна — верни JSON вида "
    "{\"action\":\"deliver_item\",\"item\":\"coffee\",\"target_person\":\"Oleg\"}. "
    "Если команда непонятна — верни {\"action\":\"unknown\"}. "
    "Отвечай только JSON, без пояснений."
)

console = Console()


# --- Запрос к vLLM ---
def _post_json(url: str, payload: dict, timeout: float) -> dict:
    """POST JSON с понятной обработкой ошибок."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"vLLM недоступен по адресу {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"vLLM вернул некорректный JSON: {exc}") from exc


def llm_parse_intent(command: str, url: str, model: str,
                     temperature: float, max_tokens: int) -> dict:
    """Попросить LLM разобрать интент команды.

    Returns:
        {action, item, target_person}
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": command},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = _post_json(url, payload, LLM_TIMEOUT_S)
    try:
        content = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Неожиданный ответ vLLM: {resp}") from exc
    # Пытаемся вытащить JSON из ответа
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Ищем подстроку {...}
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"LLM вернул не JSON: {content!r}") from exc
        raise RuntimeError(f"LLM вернул не JSON: {content!r}")


def llm_call_tool_decision(command: str, state: str, history: list[dict],
                           url: str, model: str, tools_schemas: list[dict]) -> dict:
    """Опциональный шаг: LLM решает, какой tool вызвать в текущем состоянии.

    Возвращает {tool: str, args: dict} либо {tool: None}.
    """
    messages = [
        {"role": "system", "content": (
            "Ты — оркестратор робота. В текущем состоянии нужно выбрать один "
            "из доступных инструментов для выполнения команды. "
            "Верни JSON {\"tool\":\"<name>\",\"args\":{...}}."
        )},
        {"role": "user", "content": (
            f"Команда: {command}\nСостояние: {state}\n"
            f"История: {json.dumps(history, ensure_ascii=False)}"
        )},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 256,
        "tools": tools_schemas,
        "tool_choice": "auto",
    }
    try:
        resp = _post_json(url, payload, LLM_TIMEOUT_S)
        msg = resp.get("choices", [{}])[0].get("message", {})
        if msg.get("tool_calls"):
            call = msg["tool_calls"][0]
            return {"tool": call["function"]["name"],
                    "args": json.loads(call["function"].get("arguments", "{}"))}
        # fallback: парсим контент
        content = msg.get("content", "")
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"tool": None, "reason": "no_tool_call"}
    except Exception as exc:  # noqa: BLE001
        return {"tool": None, "reason": str(exc)}


# --- Проверка доступности ---
def llm_available(url: str = DEFAULT_LLM_URL, timeout: float = 2.0) -> bool:
    """Быстрая проверка: отвечает ли vLLM."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as _:
            return True
    except Exception:  # noqa: BLE001
        return False


# --- Агент ---
class CoffeeAgent:
    """Гибридный LLM + state-machine агент."""

    def __init__(self, llm_url: str, model: str, temperature: float,
                 max_tokens: int, use_llm: bool, steer: bool):
        self.llm_url = llm_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_llm = use_llm
        self.steer = steer

    def run(self, command: str) -> dict:
        """Запуск агента по команде. Возвращает отчёт."""
        # Импорт state_machine и tools (располагаются в той же папке)
        from state_machine import CoffeeStateMachine  # type: ignore
        from tools import TOOL_SCHEMAS  # type: ignore

        # 1) Разбор интента через LLM (если включён и доступен)
        intent = None
        llm_status = "disabled"
        if self.use_llm:
            if llm_available(self.llm_url):
                try:
                    intent = llm_parse_intent(
                        command, self.llm_url, self.model,
                        self.temperature, self.max_tokens
                    )
                    llm_status = "ok"
                    console.print(f"[green]LLM разобрал интент: {intent}[/green]")
                except RuntimeError as exc:
                    llm_status = f"error: {exc}"
                    console.print(f"[yellow]LLM-разбор не удался: {exc}[/yellow]")
                    console.print("[yellow]Fallback на эвристический парсер.[/yellow]")
            else:
                llm_status = "unavailable"
                console.print(
                    f"[yellow]vLLM недоступен по адресу {self.llm_url}. "
                    f"Использую эвристический парсер.[/yellow]"
                )

        # 2) Запуск стейт-машины (она сама разбирает команду, если intent=None)
        sm = CoffeeStateMachine(mock=True)
        if intent and intent.get("action") == "deliver_item":
            # Перезаписываем команду нормализованным вариантом, чтобы
            # эвристический парсер внутри state_machine корректно вытащил
            # target_person/item.
            item = intent.get("item", "coffee")
            person = intent.get("target_person", "Oleg")
            normalized = f"принеси {item} {person}".lower()
            report = sm.run(normalized)
        elif intent and intent.get("action") == "unknown":
            console.print("[red]LLM не понял команду. Отказ.[/red]")
            report = {"ok": False, "reason": "llm_unknown_command",
                      "intent": intent, "final_state": "IDLE"}
        else:
            report = sm.run(command)

        report["llm"] = {
            "enabled": self.use_llm,
            "status": llm_status,
            "intent": intent,
            "steer": self.steer,
            "url": self.llm_url,
            "model": self.model,
        }
        return report


# --- CLI ---
@click.command()
@click.option("--command", "-c", required=True, help="Команда, например «принеси кофе Олегу».")
@click.option("--llm", "llm_choice", type=click.Choice(["qwen-local", "none"]),
              default="qwen-local", help="Какой LLM использовать.")
@click.option("--steer", is_flag=True, default=False,
              help="Давать LLM выбирать tool на каждом шаге (экспериментально).")
@click.option("--url", default=DEFAULT_LLM_URL, help="URL vLLM endpoint.")
@click.option("--model", default=DEFAULT_MODEL, help="Имя модели.")
@click.option("--temperature", default=DEFAULT_TEMPERATURE, type=float)
@click.option("--max-tokens", default=DEFAULT_MAX_TOKENS, type=int)
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Куда сохранить JSON-отчёт.")
def main(command: str, llm_choice: str, steer: bool, url: str, model: str,
         temperature: float, max_tokens: int, output: str | None) -> None:
    """LLM-агент (vLLM Qwen2.5-7B-Instruct) + fallback на стейт-машину."""
    console.print("[bold cyan]COFFEE orchestrator — LLM agent[/bold cyan]")
    use_llm = (llm_choice == "qwen-local")
    if use_llm:
        console.print(f"Endpoint vLLM: {url}")
        console.print(f"Модель: {model}, temperature={temperature}, max_tokens={max_tokens}")
    else:
        console.print("[yellow]LLM отключён — используется стейт-машина с "
                      "эвристическим парсером.[/yellow]")

    try:
        agent = CoffeeAgent(url, model, temperature, max_tokens, use_llm, steer)
        report = agent.run(command)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Непредвиденная ошибка: {exc}[/red]")
        console.print_exception()
        sys.exit(1)

    # Сводная таблица
    table = Table(title="Итог работы агента")
    table.add_column("Параметр")
    table.add_column("Значение", justify="right")
    table.add_row("Успех", "✅" if report.get("ok") else "❌")
    table.add_row("Причина", report.get("reason", "—"))
    table.add_row("Финальное состояние", report.get("final_state", "—"))
    table.add_row("Затрачено, сек", str(report.get("elapsed_s", "—")))
    llm_block = report.get("llm", {})
    table.add_row("LLM статус", llm_block.get("status", "—"))
    table.add_row("LLM intent", json.dumps(llm_block.get("intent"), ensure_ascii=False))
    console.print(table)

    if output:
        Path(output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        console.print(f"[green]Отчёт сохранён в {output}[/green]")


if __name__ == "__main__":
    main()
