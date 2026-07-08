"""config.py — конфигурация проекта G1 EDU через YAML.

Загружает ``config/default.yaml`` (либо путь, переданный через переменную
окружения ``G1_CONFIG``), предоставляет единый объект :class:`Config` со
строгой типизацией полей (dataclass). Не зависит от pydantic — pydantic
используется только в :mod:`common.state` для структур данных.

Пример::

    from common.config import load_config
    cfg = load_config()
    print(cfg.transport.endpoints["voice_command"])
"""

from __future__ import annotations

# --- Импорты ---
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

# --- Константы ---
# Базовый путь к корню проекта (cv/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
ENV_CONFIG_VAR = "G1_CONFIG"

console = Console()


# --- Dataclass'ы конфигурации ---
@dataclass
class TransportConfig:
    """Конфигурация транспортной шины ZeroMQ."""
    zmq_context: int = 1
    endpoints: dict[str, str] = field(default_factory=dict)


@dataclass
class QdrantConfig:
    """Параметры подключения к Qdrant (векторная БД для RAG анекдотов)."""
    url: str = "http://localhost:6333"
    collection: str = "jokes_1986"


@dataclass
class VLLMConfig:
    """Параметры локального vLLM-сервера (Qwen2.5-7B-Instruct)."""
    url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class TTSConfig:
    """Параметры TTS (клон голоса С. Бурунова)."""
    base_model: str = "xtts_v2"
    checkpoint: str = "./checkpoints/burunov.pt"
    sample_rate: int = 22050
    language: str = "ru"


@dataclass
class CVConfig:
    """Параметры CV (YOLO + face recognition)."""
    cup_model: str = "./coffee/detection/weights/best.pt"
    cup_confidence: float = 0.5
    face_reference: str = "./data/oleg_face.jpg"
    face_tolerance: float = 0.6


@dataclass
class RobotConfig:
    """Параметры робота G1 EDU Ultimate D (из ТЗ)."""
    max_speed: float = 2.0             # м/с — макс. скорость G1
    carry_speed: float = 0.6           # м/с — скорость с грузом (кофе)
    approach_distance: float = 0.5     # м — дистанция остановки перед стаканом
    handover_distance: float = 0.7     # м — дистанция передачи Олегу
    grasp_force_min: int = 30          # г — мин. порог силы хвата
    grasp_force_max: int = 80          # г — макс. порог силы хвата
    anti_crush_force: int = 200        # г — мгновенный разжим
    deformation_force: int = 150       # г — порог деформации стакана
    handover_release_force: int = 20   # г — стакан принят


@dataclass
class Config:
    """Полная конфигурация проекта."""
    transport: TransportConfig = field(default_factory=TransportConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    source_path: str = ""


# --- Функции ---
def _build_config(raw: dict[str, Any]) -> Config:
    """Собирает строгий :class:`Config` из «сырого» словаря YAML.

    Неизвестные ключи игнорируются (forward-совместимость).
    """
    transport_raw = raw.get("transport", {}) or {}
    cfg = Config(
        transport=TransportConfig(
            zmq_context=int(transport_raw.get("zmq_context", 1)),
            endpoints=dict(transport_raw.get("endpoints", {})),
        ),
        qdrant=QdrantConfig(**(raw.get("qdrant", {}) or {})),
        vllm=VLLMConfig(**(raw.get("vllm", {}) or {})),
        tts=TTSConfig(**(raw.get("tts", {}) or {})),
        cv=CVConfig(**(raw.get("cv", {}) or {})),
        robot=RobotConfig(**(raw.get("robot", {}) or {})),
    )
    return cfg


def load_config(path: str | Path | None = None) -> Config:
    """Загружает конфигурацию из YAML-файла.

    Порядок поиска пути:
    1. Явный аргумент ``path``.
    2. Переменная окружения ``G1_CONFIG``.
    3. ``config/default.yaml`` в корне проекта.

    При отсутствии файла или ошибке парсинга — печатается понятное сообщение
    на русском, и возвращается конфигурация по умолчанию (значения из
    dataclass'ов). Это позволяет запускать демо без YAML.
    """
    candidate = (
        Path(path).expanduser().resolve()
        if path
        else (
            Path(os.environ[ENV_CONFIG_VAR]).expanduser().resolve()
            if os.environ.get(ENV_CONFIG_VAR)
            else DEFAULT_CONFIG_PATH
        )
    )

    if not candidate.exists():
        console.print(
            f"[yellow]Внимание: файл конфигурации не найден: {candidate}\n"
            f"Используются значения по умолчанию. Создайте {DEFAULT_CONFIG_PATH} "
            f"или укажите путь через переменную окружения {ENV_CONFIG_VAR}.[/yellow]"
        )
        cfg = _build_config({})
        cfg.source_path = str(candidate)
        return cfg

    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        console.print(
            "[red]Ошибка: PyYAML не установлен. Установите: pip install pyyaml[/red]"
        )
        raise SystemExit(2) from exc

    try:
        with candidate.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        console.print(
            f"[red]Ошибка парсинга YAML ({candidate}):[/red]\n{exc}"
        )
        raise SystemExit(2) from exc

    cfg = _build_config(raw)
    cfg.source_path = str(candidate)
    return cfg


# --- CLI для отладки ---
def main() -> None:
    """CLI: печатает загруженную конфигурацию. Удобно для отладки."""
    import json
    from dataclasses import asdict

    cfg = load_config()
    console.print(f"[cyan]Источник:[/cyan] {cfg.source_path or '<default>'}")
    data = asdict(cfg)
    # Не печатаем source_path в JSON-дампе (он добавлен сверху)
    data.pop("source_path", None)
    console.print_json(json.dumps(data, ensure_ascii=False))


if __name__ == "__main__":
    main()
