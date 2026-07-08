"""logger.py — настроенный логгер на базе rich с поддержкой файла.

Предоставляет фабрику :func:`get_logger`, которая возвращает
``logging.Logger`` с цветным выводом в консоль (через ``rich.logging``) и
опциональным дублированием в файл.

Пример::

    from common.logger import get_logger
    log = get_logger("voice")
    log.info("Запущен модуль TTS")
"""

from __future__ import annotations

# --- Импорты ---
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

# --- Константы ---
# Базовый путь к корню проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s"
DEFAULT_LOG_DATEFMT = "[%Y-%m-%d %H:%M:%S]"
# Максимальный размер файла лога — 5 МБ, храним 3 резервных копии
MAX_LOG_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3

# Кэш созданных логгеров, чтобы не переинициализировать при повторном вызове
_LOGGERS: dict[str, logging.Logger] = {}


# --- Функции ---
def get_logger(
    name: str = "g1",
    *,
    level: int = logging.INFO,
    log_dir: str | Path | None = None,
    log_file: str | None = None,
    file_level: int | None = None,
    to_file: bool = False,
) -> logging.Logger:
    """Создаёт или возвращает кэшированный логгер.

    Args:
        name: Имя логгера (обычно имя модуля: ``voice``, ``coffee``...).
        level: Уровень для консольного вывода.
        log_dir: Каталог для файла лога (по умолчанию ``<project>/logs``).
        log_file: Имя файла (по умолчанию ``<name>.log``).
        file_level: Уровень для файлового вывода (по умолчанию = ``level``).
        to_file: Дублировать лог в файл (по умолчанию ``False``).
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Не передаём наверх в root-логгер (избегаем дублей в stdout)
    logger.propagate = False
    logger.handlers.clear()

    # --- Консольный handler через rich ---
    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        markup=True,
        log_time_format="[%H:%M:%S]",
    )
    rich_handler.setLevel(level)
    logger.addHandler(rich_handler)

    # --- Файловый handler (опционально) ---
    if to_file:
        try:
            target_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
            target_dir.mkdir(parents=True, exist_ok=True)
            file_path = target_dir / (log_file or f"{name}.log")
            file_handler = RotatingFileHandler(
                file_path,
                maxBytes=MAX_LOG_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(file_level or level)
            file_handler.setFormatter(
                logging.Formatter(DEFAULT_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
            )
            logger.addHandler(file_handler)
        except OSError as exc:
            # Не роняем приложение из-за проблем с файлом — пишем предупреждение
            logger.warning(
                "[yellow]Не удалось открыть файл лога: %s[/yellow]", exc
            )

    _LOGGERS[name] = logger
    return logger


def reset_loggers() -> None:
    """Сбрасывает кэш логгеров (использовать только в тестах)."""
    for lg in _LOGGERS.values():
        for h in list(lg.handlers):
            try:
                h.close()
            except Exception:
                pass
            lg.removeHandler(h)
    _LOGGERS.clear()


if __name__ == "__main__":
    # Демо: запустите python common/logger.py чтобы увидеть разные уровни
    log = get_logger("demo", to_file=True)
    log.debug("Это debug-сообщение")
    log.info("Это [green]info[/green]-сообщение")
    log.warning("Это [yellow]warning[/yellow]-сообщение")
    log.error("Это [red]error[/red]-сообщение")
    print(f"Лог-файл: {DEFAULT_LOG_DIR / 'demo.log'}")
