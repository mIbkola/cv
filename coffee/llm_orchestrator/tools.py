#!/usr/bin/env python3
"""tools.py — инструменты (tools) для LLM-агента.

Каждый tool — функция-обёртка, которая внутри обращается к соответствующему
модулю (через прямой импорт или subprocess). На текущем этапе — функции с
mock-реализацией; при наличии железа они вызывают реальные модули.

Возвращает словари с понятной структурой, чтобы LLM мог разобрать результат.

Инструменты (по ТЗ):
- find_object(class_name) → {x, y, z, confidence}
- navigate_to(x, y) → bool
- grasp_with_force_feedback(target_pose) → bool
- verify_grasp() → {stable, force}
- find_person(name) → {x, y, z}
- handover() → bool
- speak(text) → None

Также предоставляет ``TOOL_SCHEMAS`` — список JSON-схем для function calling
(формат OpenAI / vLLM).
"""

from __future__ import annotations

# --- Импорты ---
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from rich.console import Console

# --- Константы ---
# Базовый путь к корню модуля coffee/
COFFEE_ROOT = Path(__file__).resolve().parent.parent
DETECTION_DIR = COFFEE_ROOT / "detection"
MANIPULATION_DIR = COFFEE_ROOT / "manipulation"
LOCOMOTION_DIR = COFFEE_ROOT / "locomotion"

console = Console()


# --- Утилиты ---
def _run_cli_script(script_path: Path, args: list[str],
                    timeout: float = 60.0) -> dict:
    """Запускает click-скрипт как subprocess и возвращает последний JSON
    из stdout (если есть) или {'ok': bool, 'stdout': ...}.
    """
    cmd = [sys.executable, str(script_path), *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "timeout", "script": str(script_path)}
    except FileNotFoundError as exc:
        return {"ok": False, "reason": f"script_not_found: {exc}"}
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    # Пытаемся вытащить последнюю строку-JSON из stdout
    last_json: dict[str, Any] | None = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                last_json = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if last_json is not None:
        return last_json
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": stdout[-500:],
        "stderr": stderr[-500:],
    }


# --- Инструменты (mock + реальный fallback) ---
def find_object(class_name: str = "cup") -> dict:
    """Найти объект указанного класса в кадре через CV.

    Returns:
        {"found": bool, "class_name": str,
         "xyz_camera": [x,y,z]|None, "source": str}
    """
    console.print(f"[cyan][tool] find_object(class_name={class_name!r})[/cyan]")
    if class_name.lower() in ("cup", "coffee", "стакан", "кофе"):
        # Пытаемся запустить infer.py в mock-режиме если нет камеры
        # Здесь — mock-ответ (в реальной интеграции вызывается через ZeroMQ
        # к агенту-детектору, который держит камеру открытой).
        return {
            "found": True,
            "class_name": class_name,
            "xyz_camera": [0.18, -0.05, 0.62],
            "confidence": 0.91,
            "source": "mock",
        }
    return {"found": False, "class_name": class_name, "source": "mock"}


def navigate_to(x: float, y: float, speed: float = 0.6) -> dict:
    """Доехать до точки (x, y) через SportClient + VFH.

    Returns:
        {"ok": bool, "final_pos": [x,y], "distance_to_target": float}
    """
    console.print(f"[cyan][tool] navigate_to(x={x}, y={y}, speed={speed})[/cyan]")
    return {
        "ok": True,
        "final_pos": [x, y],
        "distance_to_target": 0.42,
        "speed_m_s": speed,
        "source": "mock",
    }


def grasp_with_force_feedback(target_pose: list[float]) -> dict:
    """Хватка стакана с контролем силы.

    Args:
        target_pose: [x, y, z] в системе базы руки.

    Returns:
        {"ok": bool, "final_force_g": float, "reason": str|None}
    """
    console.print(f"[cyan][tool] grasp_with_force_feedback(target_pose={target_pose})[/cyan]")
    return {
        "ok": True,
        "final_force_g": 52.3,
        "final_position": target_pose,
        "finger_openness": 0.35,
        "source": "mock",
    }


def verify_grasp() -> dict:
    """Проверить стабильность хвата (поднять на 5 см, мониторить силу).

    Returns:
        {"stable": bool, "force_g": float, "max_delta_pct": float}
    """
    console.print("[cyan][tool] verify_grasp()[/cyan]")
    return {
        "stable": True,
        "force_g": 54.1,
        "max_delta_pct": 0.08,
        "source": "mock",
    }


def find_person(name: str = "Oleg") -> dict:
    """Найти человека по имени (через face_id).

    Returns:
        {"found": bool, "name": str, "xyz_camera": [x,y,z]|None}
    """
    console.print(f"[cyan][tool] find_person(name={name!r})[/cyan]")
    if name.lower() in ("oleg", "олег"):
        return {
            "found": True,
            "name": name,
            "xyz_camera": [0.30, 0.10, 1.10],
            "confidence": 0.93,
            "source": "mock",
        }
    return {"found": False, "name": name, "source": "mock"}


def handover() -> dict:
    """Передача стакана: поднять на уровень груди, дождаться приёма.

    Returns:
        {"ok": bool, "reason": str, "wait_s": float}
    """
    console.print("[cyan][tool] handover()[/cyan]")
    return {
        "ok": True,
        "reason": "accepted",
        "wait_s": 2.4,
        "force_final_g": 5.0,
        "source": "mock",
    }


def speak(text: str) -> dict:
    """Произнести текст через TTS (модуль VOICE).

    Returns:
        {"ok": bool, "audio_path": str|None}
    """
    console.print(f"[magenta][tool] speak(text={text!r})[/magenta]")
    # Интеграция с voice/tts/infer.py — если есть
    return {"ok": True, "audio_path": None, "source": "mock"}


# --- Реестр инструментов ---
TOOL_REGISTRY: dict[str, Callable[..., dict]] = {
    "find_object": find_object,
    "navigate_to": navigate_to,
    "grasp_with_force_feedback": grasp_with_force_feedback,
    "verify_grasp": verify_grasp,
    "find_person": find_person,
    "handover": handover,
    "speak": speak,
}


# --- JSON-схемы для function calling (формат OpenAI/vLLM) ---
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "find_object",
            "description": "Найти объект указанного класса в кадре с камеры робота. "
                           "Возвращает 3D-координаты в системе камеры.",
            "parameters": {
                "type": "object",
                "properties": {
                    "class_name": {
                        "type": "string",
                        "description": "Класс объекта: 'cup' (стакан), 'person' (человек) и т.д.",
                        "default": "cup",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "navigate_to",
            "description": "Доехать до точки (x, y) в системе координат карты.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X координата, метры"},
                    "y": {"type": "number", "description": "Y координата, метры"},
                    "speed": {"type": "number", "description": "Скорость, м/с (0.5–0.8 с грузом)",
                              "default": 0.6},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grasp_with_force_feedback",
            "description": "Взять стакан с контролем силы (force-guided grasp). "
                           "target_pose — 3D-координаты стакана в системе базы руки.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_pose": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "[x, y, z] в метрах, система базы руки",
                    },
                },
                "required": ["target_pose"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_grasp",
            "description": "Проверить стабильность хвата (поднять на 5 см, мониторить силу).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_person",
            "description": "Найти человека по имени (через face_id). Возвращает 3D-позицию лица.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Имя человека", "default": "Oleg"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "handover",
            "description": "Передать стакан: поднять на уровень груди, дождаться приёма "
                           "(сила на пальцах < 20 г).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "speak",
            "description": "Произнести текст вслух через TTS.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Текст для произнесения"},
                },
                "required": ["text"],
            },
        },
    },
]


def call_tool(name: str, **kwargs) -> dict:
    """Вызвать инструмент по имени. Бросает ValueError если неизвестен."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        raise ValueError(f"Неизвестный инструмент: {name}. "
                         f"Доступные: {list(TOOL_REGISTRY)}")
    try:
        return fn(**kwargs)
    except TypeError as exc:
        return {"ok": False, "reason": f"bad_arguments: {exc}"}


# --- CLI для отладки отдельного tool ---
if __name__ == "__main__":
    import click

    @click.command()
    @click.argument("tool_name")
    @click.option("--args", "-a", default="{}",
                  help="JSON с аргументами инструмента.")
    def _cli(tool_name: str, args: str) -> None:
        """Быстрый запуск инструмента для отладки.

        Пример::

            python tools.py find_object --args '{"class_name":"cup"}'
            python tools.py navigate_to --args '{"x":1.5,"y":2.0}'
        """
        try:
            kwargs = json.loads(args)
        except json.JSONDecodeError as exc:
            console.print(f"[red]Некорректный JSON аргументов: {exc}[/red]")
            sys.exit(2)
        try:
            result = call_tool(tool_name, **kwargs)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            sys.exit(2)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    _cli()
