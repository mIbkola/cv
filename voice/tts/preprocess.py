"""Предобработка аудио/видео с голосом для обучения TTS (клон голоса Бурунова).

Скрипт принимает папку с «сырыми» аудио/видео файлами (mp3, wav, mp4),
выполняет следующие шаги:
    1. Извлечение аудиодорожки из видео через ffmpeg.
    2. Нарезка на фразы длиной 3–15 секунд через silero-vad.
    3. Нормализация: 22050 Гц, моно, 16-bit PCM.
    4. Распознавание текста через Whisper, сохранение .txt рядом с .wav.
    5. Складывание результата в processed/.

Скрипт НЕ качает файлы из YouTube — пользователь кладёт исходники в raw/ вручную.

Запуск:
    python preprocess.py --raw ../data/voice_samples/raw \
                         --out  ../data/voice_samples/processed \
                         --whisper-model base
"""

from __future__ import annotations

# --- Импорты ---------------------------------------------------------------
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

# --- Константы -------------------------------------------------------------

# Целевые параметры звука для XTTS v2.
TARGET_SR: int = 22050          # частота дискретизации, Гц
TARGET_CHANNELS: int = 1        # моно
TARGET_SAMPLE_WIDTH: int = 2    # 16-bit = 2 байта

# Параметры нарезки фраз (в секундах).
MIN_PHRASE_SEC: float = 3.0
MAX_PHRASE_SEC: float = 15.0

# Поддерживаемые расширения исходников.
AUDIO_EXTS: Tuple[str, ...] = (".mp3", ".wav", ".m4a", ".flac", ".ogg")
VIDEO_EXTS: Tuple[str, ...] = (".mp4", ".mkv", ".mov", ".avi", ".webm")

# Путь по умолчанию к models (для Whisper / silero).
DEFAULT_WHISPER_MODEL: str = "base"

console = Console()


# --- Вспомогательные функции ----------------------------------------------

def _check_tool(name: str, hint: str) -> None:
    """Проверяет наличие внешней утилиты в PATH, иначе падает с понятной ошибкой."""
    if shutil.which(name) is None:
        console.print(f"[bold red]Ошибка:[/] внешняя утилита '{name}' не найдена в PATH.")
        console.print(f"   Подсказка: {hint}")
        sys.exit(1)


def _check_python_pkg(import_name: str, pip_name: str, hint: str = "") -> None:
    """Проверяет наличие Python-пакета, иначе падает с понятной ошибкой."""
    try:
        __import__(import_name)
    except ImportError:
        console.print(f"[bold red]Ошибка:[/] Python-пакет '{pip_name}' не установлен.")
        console.print(f"   Установите: pip install {pip_name}")
        if hint:
            console.print(f"   {hint}")
        sys.exit(1)


def is_video(path: Path) -> bool:
    """Возвращает True, если файл — видео по расширению."""
    return path.suffix.lower() in VIDEO_EXTS


def is_audio(path: Path) -> bool:
    """Возвращает True, если файл — аудио по расширению."""
    return path.suffix.lower() in AUDIO_EXTS


def extract_audio_from_video(
    video_path: Path,
    out_path: Path,
    sr: int = TARGET_SR,
) -> Path:
    """Извлекает аудиодорожку из видео через ffmpeg.

    Аргументы:
        video_path: путь к видеофайлу.
        out_path:   путь к wav-файлу, куда сохранить аудио.
        sr:         целевая частота дискретизации.

    Возвращает путь к полученному wav.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                 # без видео
        "-ac", "1",            # моно
        "-ar", str(sr),        # частота
        "-acodec", "pcm_s16le",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        console.print(
            f"[bold red]ffmpeg не смог извлечь аудио из {video_path}[/]\n"
            f"{proc.stderr.decode('utf-8', errors='ignore')}"
        )
        sys.exit(2)
    return out_path


def convert_to_wav(src: Path, out_path: Path, sr: int = TARGET_SR) -> Path:
    """Перегоняет любой аудиофайл в wav 22050 Гц моно 16-bit через ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ac", "1",
        "-ar", str(sr),
        "-acodec", "pcm_s16le",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        console.print(
            f"[bold red]ffmpeg не смог конвертировать {src}[/]\n"
            f"{proc.stderr.decode('utf-8', errors='ignore')}"
        )
        sys.exit(2)
    return out_path


def load_vad_model():
    """Загружает silero-vad. Падает с понятной ошибкой при отсутствии."""
    try:
        # silero-vad >= 5.x экспортирует load_silero_vad
        from silero_vad import load_silero_vad, get_speech_timestamps
        return load_silero_vad(), get_speech_timestamps
    except ImportError:
        try:
            # Старый способ через torch.hub
            import torch
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            get_speech_timestamps = utils[0]
            return model, get_speech_timestamps
        except Exception as exc:  # noqa: BLE001
            console.print("[bold red]Ошибка:[/] не удалось загрузить silero-vad.")
            console.print(f"   Причина: {exc}")
            console.print("   Установите: pip install silero-vad")
            sys.exit(3)


def load_whisper(model_name: str):
    """Загружает Whisper. Падает с понятной ошибкой при отсутствии."""
    try:
        import whisper
    except ImportError:
        console.print("[bold red]Ошибка:[/] openai-whisper не установлен.")
        console.print("   Установите: pip install openai-whisper")
        sys.exit(4)
    try:
        return whisper.load_model(model_name)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Ошибка:[/] не удалось загрузить Whisper-модель '{model_name}'.")
        console.print(f"   Причина: {exc}")
        sys.exit(5)


def slice_wav_by_timestamps(
    src_wav: Path,
    timestamps: List[dict],
    out_dir: Path,
    sr: int = TARGET_SR,
) -> List[Path]:
    """Нарезает исходный wav по таймстампам silero-vad.

    Аргументы:
        src_wav:    путь к исходному wav (моно, 22050 Гц).
        timestamps: список словарей {'start': samples, 'end': samples}.
        out_dir:    куда складывать фразы.
        sr:         частота дискретизации.

    Возвращает список путей к нарезанным wav-файлам.
    """
    import torch
    import torchaudio

    waveform, sr_orig = torchaudio.load(str(src_wav))  # (channels, samples)
    if sr_orig != sr:
        waveform = torchaudio.functional.resample(waveform, sr_orig, sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    out_paths: List[Path] = []
    for idx, ts in enumerate(timestamps):
        start_s = ts["start"] / sr
        end_s = ts["end"] / sr
        dur = end_s - start_s
        # Отбрасываем слишком короткие / слишком длинные фразы.
        if dur < MIN_PHRASE_SEC or dur > MAX_PHRASE_SEC:
            continue
        chunk = waveform[:, ts["start"]:ts["end"]]
        out_path = out_dir / f"{src_wav.stem}_{idx:04d}.wav"
        torchaudio.save(
            str(out_path), chunk, sr,
            encoding="PCM_S", bits_per_sample=16,
        )
        out_paths.append(out_path)
    return out_paths


def transcribe_chunk(whisper_model, wav_path: Path) -> str:
    """Распознаёт текст в wav-файле через Whisper, возвращает очищенный текст."""
    result = whisper_model.transcribe(
        str(wav_path),
        language="ru",
        task="transcribe",
        verbose=False,
    )
    text: str = result.get("text", "").strip()
    return " ".join(text.split())  # нормализуем пробелы


# --- Основная логика ------------------------------------------------------

def process_one_file(
    src_path: Path,
    tmp_dir: Path,
    out_dir: Path,
    vad_model,
    get_speech_timestamps,
    whisper_model,
    sr: int = TARGET_SR,
) -> int:
    """Полный пайплайн для одного исходного файла.

    Возвращает количество нарезанных фраз.
    """
    # 1) Достаём / конвертируем в промежуточный wav.
    tmp_wav = tmp_dir / f"{src_path.stem}.wav"
    if is_video(src_path):
        console.print(f"   [cyan]Извлечение аудио из видео[/] {src_path.name}...")
        extract_audio_from_video(src_path, tmp_wav, sr=sr)
    elif is_audio(src_path):
        console.print(f"   [cyan]Конвертация аудио[/] {src_path.name}...")
        convert_to_wav(src_path, tmp_wav, sr=sr)
    else:
        console.print(f"   [yellow]Пропуск неподдерживаемого файла:[/] {src_path.name}")
        return 0

    # 2) Нарезаем через silero-vad.
    import torch
    import torchaudio

    wav, sr_orig = torchaudio.load(str(tmp_wav))
    if sr_orig != sr:
        wav = torchaudio.functional.resample(wav, sr_orig, sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav_1d = wav.squeeze(0)

    # silero-vad ожидает 16 кГц для своей внутренней модели.
    vad_sr = 16000
    if sr != vad_sr:
        wav_vad = torchaudio.functional.resample(wav, sr, vad_sr).squeeze(0)
    else:
        wav_vad = wav_1d

    timestamps = get_speech_timestamps(
        wav_vad, vad_model,
        return_seconds=False,
        sampling_rate=vad_sr,
        min_silence_duration_ms=300,
        speech_pad_ms=200,
    )

    # 3) Сохраняем нарезанные фразы.
    chunk_paths = slice_wav_by_timestamps(tmp_wav, timestamps, out_dir, sr=sr)
    console.print(f"   [green]Нарезано фраз:[/] {len(chunk_paths)}")

    # 4) Транскрипция Whisper + .txt рядом с .wav.
    for cp in chunk_paths:
        text = transcribe_chunk(whisper_model, cp)
        txt_path = cp.with_suffix(".txt")
        txt_path.write_text(text, encoding="utf-8")

    # Чистим временный файл.
    try:
        tmp_wav.unlink()
    except OSError:
        pass

    return len(chunk_paths)


# --- CLI -------------------------------------------------------------------

@click.command()
@click.option(
    "--raw", "raw_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=Path("../data/voice_samples/raw"),
    show_default=True,
    help="Папка с исходными аудио/видео файлами (mp3, wav, mp4, ...).",
)
@click.option(
    "--out", "out_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("../data/voice_samples/processed"),
    show_default=True,
    help="Куда складывать нарезанные wav + txt.",
)
@click.option(
    "--whisper-model",
    type=str,
    default=DEFAULT_WHISPER_MODEL,
    show_default=True,
    help="Размер модели Whisper: tiny|base|small|medium|large.",
)
@click.option(
    "--sr",
    type=int,
    default=TARGET_SR,
    show_default=True,
    help="Целевая частота дискретизации, Гц.",
)
def main(raw_dir: Path, out_dir: Path, whisper_model: str, sr: int) -> None:
    """Предобработка аудио/видео для fine-tune TTS.

    Скрипт берёт файлы из --raw, нарезает на фразы 3–15 сек, нормализует
    (моно, 22050 Гц, 16-bit), распознаёт текст через Whisper и складывает
    результат в --out (wav + txt парами).
    """
    console.print("[bold blue]== Предобработка голоса для TTS (Бурунов) ==[/]")

    # --- Проверки зависимостей ---
    _check_tool("ffmpeg", "установите ffmpeg, см. https://ffmpeg.org/download.html")
    _check_python_pkg("torch", "torch", "pip install torch torchaudio")
    _check_python_pkg("torchaudio", "torchaudio")
    _check_python_pkg("whisper", "openai-whisper")
    _check_python_pkg("silero_vad", "silero-vad")

    # --- Поиск исходников ---
    src_files: List[Path] = []
    for p in sorted(raw_dir.iterdir()):
        if p.is_file() and (is_audio(p) or is_video(p)):
            src_files.append(p)

    if not src_files:
        console.print(
            f"[bold yellow]Внимание:[/] в папке {raw_dir} нет поддерживаемых файлов "
            "(mp3/wav/mp4/...). Сначала положите туда исходники."
        )
        sys.exit(0)

    console.print(f"Найдено исходников: [bold]{len(src_files)}[/]")
    console.print(f"Целевая частота:    [bold]{sr} Гц[/], моно, 16-bit")

    # --- Подготовка папок ---
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # --- Загрузка моделей ---
    console.print("Загрузка silero-vad...")
    vad_model, get_speech_timestamps = load_vad_model()
    console.print(f"Загрузка Whisper '{whisper_model}'...")
    whisper_mdl = load_whisper(whisper_model)

    # --- Обработка ---
    total_chunks = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Обработка файлов...", total=len(src_files))
        for src in src_files:
            console.print(f"\n[bold green]▶ {src.name}[/]")
            try:
                n = process_one_file(
                    src, tmp_dir, out_dir,
                    vad_model, get_speech_timestamps, whisper_mdl,
                    sr=sr,
                )
                total_chunks += n
            except Exception as exc:  # noqa: BLE001
                console.print(f"[bold red]Ошибка при обработке {src.name}:[/] {exc}")
            progress.advance(task)

    # Чистим tmp.
    shutil.rmtree(tmp_dir, ignore_errors=True)

    console.print(
        f"\n[bold green]Готово.[/] Всего нарезано фраз: {total_chunks}. "
        f"Результат в: {out_dir}"
    )


if __name__ == "__main__":
    main()
