"""state.py — общие структуры данных для обмена между модулями.

Все структуры — pydantic v2-модели (сериализуются в JSON для передачи
через ZeroMQ). Если pydantic не установлен — используется легковесный
fallback на dataclass + ручная JSON-сериализация (через ``model_dump``
имитируется).

Структуры:
- :class:`Detection3D` — результат детекции объекта (YOLO + depth).
- :class:`FaceMatch` — распознанное лицо (для Олега и др.).
- :class:`GraspResult` — результат хватания стакана.
- :class:`NavigationGoal` — цель навигации (для локомоции).
- :class:`VoiceCommand` — голосовая команда от ASR к оркестратору.

Каждая модель имеет метод ``to_json`` / ``from_json`` для удобства
передачи по сети.
"""

from __future__ import annotations

# --- Импорты ---
import json
from typing import Any, Optional

# --- Константы ---
# Признак, что pydantic v2 доступен
try:
    from pydantic import BaseModel, Field, ConfigDict  # type: ignore
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False
    BaseModel = object  # type: ignore[assignment,misc]


# --- Fallback (если pydantic не установлен) ---
class _FallbackModel:
    """Минимальный заглушка-pydantic на базе dataclass-логики.

    Поддерживает __init__ с kw-only полями, model_dump, model_validate,
    to_json, from_json. Не валидирует типы — только сериализует.
    """

    _fields: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        for name, default in self._fields.items():
            setattr(self, name, kwargs.get(name, default))

    def model_dump(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self._fields}

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> "_FallbackModel":
        return cls(**{k: v for k, v in data.items() if k in cls._fields})

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, raw: str | bytes) -> "_FallbackModel":
        return cls.model_validate(json.loads(raw))

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={getattr(self, k)!r}" for k in self._fields)
        return f"{self.__class__.__name__}({attrs})"


def _make_model(name: str, fields: dict[str, Any]) -> type:
    """Создаёт модель либо через pydantic, либо через fallback."""
    if _HAS_PYDANTIC:
        annotations: dict[str, Any] = {}
        defaults: dict[str, Any] = {}
        for fname, fdefault in fields.items():
            # Аннотацию не знаем точно в fallback-режиме, делаем Optional[Any]
            annotations[fname] = Any
            defaults[fname] = fdefault
        namespace = {"__annotations__": annotations, **defaults}
        # pydantic v2 BaseModel с динамическими полями
        return type(name, (BaseModel,), namespace)  # type: ignore[arg-type]
    else:
        # Fallback — явно задаём _fields
        cls = type(name, (_FallbackModel,), {"_fields": dict(fields)})
        return cls


# --- Модели ---
# Detection3D — результат CV-детекции (YOLO + RealSense depth)
Detection3D = _make_model(
    "Detection3D",
    {
        "class_name": "cup",
        "confidence": 0.0,
        "x": 0.0,  # метры, в СК робота
        "y": 0.0,
        "z": 0.0,
        "bbox_xyxy": [0.0, 0.0, 0.0, 0.0],  # в пикселях
        "source": "mock",  # mock | yolov8 | rtdetr
        "timestamp_ms": 0,
    },
)

# FaceMatch — распознанное лицо
FaceMatch = _make_model(
    "FaceMatch",
    {
        "name": "unknown",        # имя из базы (или "unknown")
        "distance": 1.0,          # расстояние в пространстве эмбеддингов
        "matched": False,
        "x": 0.0,                 # 3D-позиция лица в СК робота
        "y": 0.0,
        "z": 0.0,
        "confidence": 0.0,
        "timestamp_ms": 0,
    },
)

# GraspResult — итог хватания
GraspResult = _make_model(
    "GraspResult",
    {
        "success": False,
        "force_g": 0.0,            # фактическая сила в граммах
        "stable": False,           # прошёл stability_check?
        "reason": "",              # описание ошибки или "ok"
        "attempts": 0,             # число попыток
        "timestamp_ms": 0,
    },
)

# NavigationGoal — цель локомоции
NavigationGoal = _make_model(
    "NavigationGoal",
    {
        "x": 0.0,
        "y": 0.0,
        "theta": 0.0,             # ориентация в радианах
        "speed": 0.6,             # м/с
        "stop_distance": 0.5,     # м — дистанция остановки
        "reason": "approach",     # approach_cup | approach_target | return_base
        "timestamp_ms": 0,
    },
)

# VoiceCommand — голосовая команда
VoiceCommand = _make_model(
    "VoiceCommand",
    {
        "text": "",               # распознанный текст
        "intent": "",             # coffee | joke | unknown
        "params": {},             # доп. параметры (item, target_person, topic...)
        "confidence": 0.0,
        "source": "mock",         # mock | whisper | external
        "timestamp_ms": 0,
    },
)


# --- Утилита для OOP-стиля вызовов ---
def to_json(model: Any) -> str:
    """Сериализует модель (pydantic или fallback) в JSON-строку."""
    if hasattr(model, "to_json"):
        return model.to_json()
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json()
    if hasattr(model, "model_dump"):
        return json.dumps(model.model_dump(), ensure_ascii=False, default=str)
    return json.dumps(model, ensure_ascii=False, default=str)


def from_json(raw: str | bytes, model_cls: type) -> Any:
    """Десериализует JSON-строку в модель указанного класса."""
    if hasattr(model_cls, "from_json"):
        return model_cls.from_json(raw)
    if hasattr(model_cls, "model_validate_json"):
        return model_cls.model_validate_json(raw)
    data = json.loads(raw)
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls(**data)


if __name__ == "__main__":
    # Демо сериализации/десериализации
    from rich.console import Console
    console = Console()
    console.print(f"[cyan]pydantic available:[/cyan] {_HAS_PYDANTIC}")

    det = Detection3D(class_name="cup", confidence=0.92, x=0.18, y=-0.05, z=0.62)
    console.print(f"[green]Detection3D:[/green] {det!r}")
    console.print(f"[green]JSON:[/green] {to_json(det)}")
    restored = from_json(to_json(det), Detection3D)
    console.print(f"[green]Restored:[/green] {restored!r}")

    cmd = VoiceCommand(text="принеси кофе Олегу", intent="coffee",
                       params={"item": "coffee", "target_person": "Oleg"},
                       confidence=0.95)
    console.print(f"[green]VoiceCommand:[/green] {to_json(cmd)}")
