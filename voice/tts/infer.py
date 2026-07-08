"""Инференс TTS: синтез речи голосом С. Бурунова.

Скрипт принимает текст и путь к checkpoint (созданному `train.py`),
генерирует wav (22050 Гц, моно) и сохраняет в выходной файл.

Поддерживаемые режимы checkpoint:
    1. zero_shot_reference — есть папка `speaker_refs/` с эталонными wav.
       XTTS v2 клонирует голос по этой ссылке.
    2. full_finetune — полноценный fine-tune (если реализован в train.py).

Поддерживаемые языки: ru (по умолчанию), en, de, fr, es, it, pt, pl, tr, nl, ...
(полный список зависит от XTTS v2).

Запуск:
    python infer.py \
        --text "Олег, держи свой кофе, бля" \
        --checkpoint ./checkpoints/burunov \
        --out ./out.wav
"""

from __future__ import annotations

# --- Импорты ---------------------------------------------------------------
import json
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.panel import Panel

# --- Константы -------------------------------------------------------------

DEFAULT_LANG: str = "ru"
DEFAULT_SAMPLE_RATE: int = 22050
DEFAULT_BASE_MODEL: str = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_SPEAKER_REF: str = "ref_000.wav"  # какой из refs использовать по умолчанию

# Поддерживаемые XTTS v2 языки.
SUPPORTED_LANGS: tuple[str, ...] = (
    "ru", "en", "de", "fr", "es", "it", "pt", "pl", "tr", "nl",
    "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi",
)

console = Console()


# --- Утилиты ---------------------------------------------------------------

def _check_pkg(import_name: str, pip_name: str) -> None:
    """Проверяет наличие Python-пакета, иначе падает с понятной ошибкой."""
    try:
        __import__(import_name)
    except ImportError:
        console.print(f"[bold red]Ошибка:[/] Python-пакет '{pip_name}' не установлен.")
        console.print(f"   Установите: pip install {pip_name}")
        sys.exit(1)


def load_checkpoint_meta(checkpoint_dir: Path) -> Dict[str, Any]:
    """Загружает meta.json из checkpoint-папки."""
    meta_path = checkpoint_dir / "meta.json"
    if not meta_path.exists():
        console.print(
            f"[bold red]Ошибка:[/]"
            f"в checkpoint-папке {checkpoint_dir} нет meta.json. "
            "Сначала запустите train.py."
        )
        sys.exit(2)
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        console.print(f"[bold red]Ошибка:[/] meta.json повреждён: {exc}")
        sys.exit(3)


def pick_speaker_ref(checkpoint_dir: Path, meta: Dict[str, Any]) -> Path:
    """Выбирает эталонный wav для условной генерации (zero-shot режим)."""
    mode = meta.get("mode", "zero_shot_reference")
    if mode != "zero_shot_reference":
        # В режиме full_finetune референс не обязателен, но XTTS всё равно
        # требует gpt_cond_len Samples — отдаём любой, если есть.
        pass

    refs_dir = checkpoint_dir / "speaker_refs"
    if not refs_dir.exists():
        console.print(
            f"[bold red]Ошибка:[/]"
            f"в checkpoint нет папки {refs_dir.name}/ с эталонными фразами."
        )
        sys.exit(4)

    refs = sorted(refs_dir.glob("*.wav"))
    if not refs:
        console.print(
            f"[bold red]Ошибка:[/]"
            f"папка {refs_dir} не содержит .wav файлов."
        )
        sys.exit(5)
    return refs[0]


def write_wav(path: Path, audio_bytes: bytes, sr: int = DEFAULT_SAMPLE_RATE) -> None:
    """Сохраняет сырые 16-bit PCM байты в wav-файл (моно)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(audio_bytes)


# --- Синтез ----------------------------------------------------------------

def synthesize_xtts(
    text: str,
    checkpoint_dir: Path,
    speaker_ref: Path,
    language: str,
    out_path: Path,
) -> None:
    """Синтез речи через Coqui XTTS v2.

    Загружает базовую модель xtts_v2 и применяет условную генерацию
    голосом по эталонному wav.
    """
    from TTS.api import TTS
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"Устройство: [bold]{device}[/]")

    console.print("Загрузка XTTS v2...")
    try:
        tts = TTS(DEFAULT_BASE_MODEL)
        tts.to(device)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Ошибка:[/] не удалось загрузить XTTS v2: {exc}")
        console.print(
            "   Возможные причины: нет интернета для скачивания модели, "
            "закончилось место на диске, несовместимая версия TTS."
        )
        sys.exit(6)

    console.print(f"Эталонный голос: [bold]{speaker_ref.name}[/]")
    console.print(f"Язык: [bold]{language}[/]")
    console.print(f"Текст: [italic]«{text}»[/]")

    # XTTS v2: tts.tts_to_file синтезирует и сохраняет в файл.
    try:
        tts.tts_to_file(
            text=text,
            speaker_wav=str(speaker_ref),
            language=language,
            file_path=str(out_path),
            split_sentences=True,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Ошибка синтеза:[/] {exc}")
        sys.exit(7)


# --- CLI -------------------------------------------------------------------

@click.command()
@click.option(
    "--text", "text",
    type=str,
    required=True,
    help="Текст для синтеза (можно в кавычках).",
)
@click.option(
    "--checkpoint", "checkpoint_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=Path("./checkpoints/burunov"),
    show_default=True,
    help="Папка checkpoint (с meta.json и speaker_refs/).",
)
@click.option(
    "--out", "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("./out.wav"),
    show_default=True,
    help="Куда сохранить wav.",
)
@click.option(
    "--language",
    type=click.Choice(SUPPORTED_LANGS, case_sensitive=False),
    default=DEFAULT_LANG,
    show_default=True,
    help="Язык синтеза.",
)
@click.option(
    "--speaker-ref",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Переопределить эталонный wav (по умолчанию из checkpoint/speaker_refs/).",
)
def main(
    text: str,
    checkpoint_dir: Path,
    out_path: Path,
    language: str,
    speaker_ref: Optional[Path],
) -> None:
    """Синтез речи голосом С. Бурунова через XTTS v2."""
    console.print("[bold blue]== Инференс TTS (Бурунов) ==[/]")

    # Проверки зависимостей.
    _check_pkg("torch", "torch")
    _check_pkg("TTS", "TTS")  # Coqui TTS

    if not text.strip():
        console.print("[bold red]Ошибка:[/] пустой текст для синтеза.")
        sys.exit(8)

    # Загружаем meta из checkpoint.
    meta = load_checkpoint_meta(checkpoint_dir)
    console.print(
        Panel.fit(
            f"Checkpoint: {checkpoint_dir}\n"
            f"Mode: {meta.get('mode', 'unknown')}\n"
            f"Base model: {meta.get('base_model', DEFAULT_BASE_MODEL)}",
            border_style="cyan",
        )
    )

    # Выбираем эталонный wav.
    if speaker_ref is None:
        speaker_ref = pick_speaker_ref(checkpoint_dir, meta)

    # Синтез.
    synthesize_xtts(
        text=text,
        checkpoint_dir=checkpoint_dir,
        speaker_ref=speaker_ref,
        language=language,
        out_path=out_path,
    )

    console.print(f"\n[bold green]Готово.[/] Аудио сохранено: {out_path}")


if __name__ == "__main__":
    main()
