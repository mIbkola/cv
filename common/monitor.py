"""monitor.py — CLI-утилита для мониторинга ZeroMQ-топиков в реальном времени.

Подключается SUB-сокетами ко всем PUB-эндпоинтам (из ``config/default.yaml``)
и печатает входящие сообщения. Полезно при отладке интеграционных сценариев.

Запуск::

    python common/monitor.py
    python common/monitor.py --topics voice.command,cv.detection
    python common/monitor.py --json  # сырой JSON в stdout
"""

from __future__ import annotations

# --- Импорты ---
import json
import sys
import time
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Локальные импорты
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_config
from common.logger import get_logger
from common.transport import (
    Subscriber,
    TOPIC_VOICE_COMMAND,
    TOPIC_VOICE_SPEAK,
    TOPIC_COFFEE_STATE,
    TOPIC_CV_DETECTION,
    TOPIC_CV_FACE,
    ALL_TOPICS,
    is_zmq_available,
)

# --- Константы ---
# Маппинг топик → эндпоинт (берётся из конфига)
_TOPIC_TO_CONFIG_KEY = {
    TOPIC_VOICE_COMMAND: "voice_command",
    TOPIC_VOICE_SPEAK: "voice_speak",
    TOPIC_COFFEE_STATE: "coffee_state",
    TOPIC_CV_DETECTION: "cv_detection",
    TOPIC_CV_FACE: "cv_face",
}

# Цвета для разных топиков в rich-выводе
_TOPIC_COLORS = {
    TOPIC_VOICE_COMMAND: "magenta",
    TOPIC_VOICE_SPEAK: "cyan",
    TOPIC_COFFEE_STATE: "yellow",
    TOPIC_CV_DETECTION: "green",
    TOPIC_CV_FACE: "blue",
}

console = Console()
log = get_logger("monitor")


# --- Функции ---
def _build_subscribers(topics: list[str]) -> list[Subscriber]:
    """Создаёт SUB-сокеты для указанных топиков.

    Эндпоинты берутся из конфига. Для req/rep-топиков (coffee.tool_*) monitor
    не подключается — они не pub/sub.
    """
    cfg = load_config()
    subs: list[Subscriber] = []
    for topic in topics:
        cfg_key = _TOPIC_TO_CONFIG_KEY.get(topic)
        if cfg_key is None:
            log.warning(f"Топик {topic!r} не поддерживается монитором — пропуск")
            continue
        # PUB-сокеты биндят на tcp://*:<port>, SUB-сокеты подключаются на
        # tcp://localhost:<port> (или tcp://127.0.0.1:<port>).
        ep = cfg.transport.endpoints.get(cfg_key, "")
        if not ep:
            log.warning(f"Эндпоинт для топика {topic!r} не задан в конфиге — пропуск")
            continue
        sub_ep = ep.replace("*", "localhost")
        try:
            sub = Subscriber(sub_ep, topic=topic, poll_timeout_ms=200)
            subs.append(sub)
            log.info(f"Подписан на {topic!r} @ {sub_ep}")
        except Exception as exc:  # noqa: BLE001
            log.error(f"Не удалось подписаться на {topic!r} @ {sub_ep}: {exc}")
    return subs


def _format_message(env: dict[str, Any], *, json_mode: bool = False) -> str:
    """Форматирует сообщение для вывода."""
    if json_mode:
        return json.dumps(env, ensure_ascii=False, default=str)
    topic = env.get("topic", "?")
    ts = env.get("ts_ms", 0)
    payload = env.get("payload", {})
    color = _TOPIC_COLORS.get(topic, "white")
    # Сокращаем длинные payload-ы для читаемости
    payload_str = json.dumps(payload, ensure_ascii=False, default=str)
    if len(payload_str) > 240:
        payload_str = payload_str[:240] + "…"
    ts_str = time.strftime("%H:%M:%S", time.localtime(ts / 1000)) if ts else "--:--:--"
    return f"[{color}]{topic:24s}[/{color}] | {ts_str} | {payload_str}"


def _render_history(history: list[dict[str, Any]], max_rows: int = 30) -> Table:
    """rich-таблица истории сообщений."""
    t = Table(title="Monitor — последние сообщения", border_style="cyan",
              show_lines=False, expand=True)
    t.add_column("Время", style="dim", width=10)
    t.add_column("Топик", width=22)
    t.add_column("Payload", overflow="fold")
    for env in history[-max_rows:]:
        ts = env.get("ts_ms", 0)
        ts_str = time.strftime("%H:%M:%S", time.localtime(ts / 1000)) if ts else "--:--:--"
        topic = env.get("topic", "?")
        color = _TOPIC_COLORS.get(topic, "white")
        payload = env.get("payload", {})
        payload_str = json.dumps(payload, ensure_ascii=False, default=str)
        if len(payload_str) > 200:
            payload_str = payload_str[:200] + "…"
        t.add_row(ts_str, Text(topic, style=color), payload_str)
    return t


# --- CLI ---
@click.command()
@click.option("--topics", default=",".join(_TOPIC_TO_CONFIG_KEY.keys()),
              show_default=True,
              help="Список топиков через запятую для мониторинга.")
@click.option("--json", "json_mode", is_flag=True,
              help="Сырой JSON в stdout (вместо rich-таблицы).")
@click.option("--max-messages", default=0, show_default=True,
              help="Остановиться после N сообщений (0 = бесконечно).")
@click.option("--live-table", is_flag=True,
              help="Живая rich-таблица вместо поточного вывода.")
def main(topics: str, json_mode: bool, max_messages: int, live_table: bool) -> None:
    """Запускает мониторинг ZeroMQ-топиков."""
    if not is_zmq_available():
        console.print(
            "[yellow]Внимание: pyzmq не установлен — сообщения не будут "
            "получены. Установите: [cyan]pip install pyzmq[/cyan][/yellow]"
        )

    topic_list = [t.strip() for t in topics.split(",") if t.strip()]
    console.print(Panel(
        f"[bold]Мониторинг топиков:[/bold] {', '.join(topic_list)}\n"
        f"Ctrl+C — выход.",
        title="G1 ZMQ Monitor",
        border_style="cyan",
    ))

    subs = _build_subscribers(topic_list)
    if not subs:
        console.print("[red]Не создано ни одного подписчика — выход.[/red]")
        return

    history: list[dict[str, Any]] = []
    count = 0
    try:
        if live_table and not json_mode:
            with Live(_render_history(history), refresh_per_second=4,
                      console=console) as live:
                while True:
                    for sub in subs:
                        env = sub.recv_json(timeout_ms=100)
                        if env is not None:
                            history.append(env)
                            if len(history) > 100:
                                history = history[-100:]
                            count += 1
                            live.update(_render_history(history))
                            if max_messages and count >= max_messages:
                                raise KeyboardInterrupt
                    time.sleep(0.02)
        else:
            while True:
                for sub in subs:
                    env = sub.recv_json(timeout_ms=100)
                    if env is not None:
                        history.append(env)
                        if json_mode:
                            print(_format_message(env, json_mode=True))
                        else:
                            console.print(_format_message(env))
                        count += 1
                        if max_messages and count >= max_messages:
                            raise KeyboardInterrupt
                time.sleep(0.02)
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Monitor остановлен. Получено сообщений: {count}[/yellow]")
    finally:
        for sub in subs:
            sub.close()


if __name__ == "__main__":
    main()
