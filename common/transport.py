"""transport.py — обёртка над ZeroMQ для pub/sub и req/rep между модулями.

Топики (определены в ``config/default.yaml``)::

    voice.command  (pub)  — голосовые команды от ASR к оркестратору
    voice.speak    (sub)  — текст для TTS
    coffee.state   (pub)  — текущее состояние state machine
    coffee.tool_call (req) → coffee.tool_result (rep) — вызовы инструментов
    cv.detection   (pub)  — результаты детекции
    cv.face        (pub)  — распознанные лица

Классы:
- :class:`Publisher`  — PUB-сокет (один издатель на топик).
- :class:`Subscriber` — SUB-сокет (подписчик на топик).
- :class:`Requester`  — REQ-сокет (клиент req/rep).
- :class:`Replier`    — REP-сокет (сервер req/rep).

Все сообщения — JSON (UTF-8). Каждое сообщение оборачивается в конверт::

    {"topic": <topic>, "ts_ms": <epoch_ms>, "payload": <user_data>}

Если pyzmq не установлен — все классы работают в «заглушечном» режиме
(печатают в консоль, не падают). Это позволяет запускать демо без
ZeroMQ-зависимостей.
"""

from __future__ import annotations

# --- Импорты ---
import json
import time
from typing import Any, Callable, Optional

from rich.console import Console

# --- Константы ---
# Признак доступности pyzmq
try:
    import zmq  # type: ignore
    _HAS_ZMQ = True
except ImportError:  # pragma: no cover
    _HAS_ZMQ = False

# Имена топиков (один источник истины для всего проекта)
TOPIC_VOICE_COMMAND = "voice.command"
TOPIC_VOICE_SPEAK = "voice.speak"
TOPIC_COFFEE_STATE = "coffee.state"
TOPIC_COFFEE_TOOL_CALL = "coffee.tool_call"
TOPIC_COFFEE_TOOL_RESULT = "coffee.tool_result"
TOPIC_CV_DETECTION = "cv.detection"
TOPIC_CV_FACE = "cv.face"

ALL_TOPICS = (
    TOPIC_VOICE_COMMAND,
    TOPIC_VOICE_SPEAK,
    TOPIC_COFFEE_STATE,
    TOPIC_COFFEE_TOOL_CALL,
    TOPIC_COFFEE_TOOL_RESULT,
    TOPIC_CV_DETECTION,
    TOPIC_CV_FACE,
)

console = Console()


# --- Утилиты ---
def _wrap(topic: str, payload: Any) -> bytes:
    """Упаковывает payload в JSON-конверт с метаданными."""
    envelope = {
        "topic": topic,
        "ts_ms": int(time.time() * 1000),
        "payload": payload,
    }
    return json.dumps(envelope, ensure_ascii=False, default=str).encode("utf-8")


def _unwrap(data: bytes) -> dict[str, Any]:
    """Распаковывает конверт, возвращает словарь {topic, ts_ms, payload}."""
    if not data:
        return {"topic": "", "ts_ms": 0, "payload": None}
    try:
        return json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {
            "topic": "",
            "ts_ms": 0,
            "payload": None,
            "_error": f"Не удалось распаковать сообщение: {exc}",
        }


# --- Заглушка для режима без pyzmq ---
class _StubSocket:
    """Внутренняя заглушка, эмулирующая ZeroMQ-сокет на консоли.

    Не передаёт сообщения между процессами — только печатает их.
    Используется когда pyzmq не установлен, чтобы демо не падали.
    """

    def __init__(self, role: str, endpoint: str, topic: str = "") -> None:
        self.role = role
        self.endpoint = endpoint
        self.topic = topic
        console.print(
            f"[yellow][stub-ZMQ] {role} endpoint={endpoint} "
            f"topic={topic or '-'} (pyzmq не установлен, "
            f"сообщения печатаются в консоль)[/yellow]"
        )

    def send(self, data: bytes, flags: int = 0) -> None:
        env = _unwrap(data)
        console.print(
            f"[yellow][stub-ZMQ] → {self.topic or self.role}[/yellow] "
            f"{json.dumps(env.get('payload'), ensure_ascii=False)[:200]}"
        )

    def recv(self, flags: int = 0) -> bytes:
        # В stub-режиме нельзя получать сообщения; вернём пустой конверт
        return _unwrap(b"")

    def close(self) -> None:
        pass

    def setsockopt(self, *args: Any, **kwargs: Any) -> None:
        pass

    def bind(self, addr: str) -> None:
        pass

    def connect(self, addr: str) -> None:
        pass


# --- Публичные классы ---
class Publisher:
    """Издатель ZeroMQ (PUB-сокет).

    Использует bind (один издатель на топик). Для подписчиков — connect.
    """

    def __init__(self, endpoint: str, topic: str, *, context: Any = None) -> None:
        self.endpoint = endpoint
        self.topic = topic
        self._sock: Any = None
        self._ctx: Any = None
        self._stub: Optional[_StubSocket] = None
        self._init_socket(context)

    def _init_socket(self, context: Any) -> None:
        if not _HAS_ZMQ:
            self._stub = _StubSocket("PUB", self.endpoint, self.topic)
            return
        self._ctx = context or zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        # Лёгкое сглаживание при старте (drop старых подписчиков)
        self._sock.setsockopt(zmq.LINGER, 100)
        self._sock.bind(self.endpoint)

    def publish(self, payload: Any) -> None:
        """Публикует payload в топик."""
        if self._stub is not None:
            self._stub.send(_wrap(self.topic, payload))
            return
        assert self._sock is not None
        # PUB в ZMQ: первая часть — топик, вторая — конверт
        self._sock.send_multipart([
            self.topic.encode("utf-8"),
            _wrap(self.topic, payload),
        ])

    def close(self) -> None:
        """Закрывает сокет. Вызывать при завершении процесса."""
        if self._sock is not None:
            self._sock.close(linger=100)
        if self._stub is not None:
            self._stub.close()


class Subscriber:
    """Подписчик ZeroMQ (SUB-сокет).

    Использует connect к издателю. Можно подписаться на подмножество топиков
    (по умолчанию — все от данного endpoint).
    """

    def __init__(
        self,
        endpoint: str,
        topic: str = "",
        *,
        context: Any = None,
        poll_timeout_ms: int = 1000,
    ) -> None:
        self.endpoint = endpoint
        self.topic = topic
        self.poll_timeout_ms = poll_timeout_ms
        self._sock: Any = None
        self._ctx: Any = None
        self._stub: Optional[_StubSocket] = None
        self._init_socket(context)

    def _init_socket(self, context: Any) -> None:
        if not _HAS_ZMQ:
            self._stub = _StubSocket("SUB", self.endpoint, self.topic)
            return
        self._ctx = context or zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        # Подписка. Пустой префикс = все топики.
        self._sock.setsockopt(zmq.SUBSCRIBE, self.topic.encode("utf-8"))
        self._sock.setsockopt(zmq.LINGER, 100)
        self._sock.connect(self.endpoint)

    def recv_json(self, timeout_ms: int | None = None) -> Optional[dict[str, Any]]:
        """Получает одно сообщение. Возвращает ``None`` по таймауту."""
        if self._stub is not None:
            return None
        assert self._sock is not None
        timeout = timeout_ms if timeout_ms is not None else self.poll_timeout_ms
        if not _has_event(self._sock, timeout):
            return None
        try:
            _topic, body = self._sock.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        return _unwrap(body)

    def consume(
        self,
        callback: Callable[[dict[str, Any]], None],
        *,
        max_messages: int = 0,
        idle_sleep: float = 0.05,
    ) -> None:
        """Бесконечный цикл приёма сообщений с вызовом callback.

        Args:
            callback: Функция, вызывается с распакованным конвертом для
                каждого сообщения.
            max_messages: Если > 0 — остановится после этого числа сообщений.
            idle_sleep: Спать, когда нет сообщений (для снижения CPU).
        """
        count = 0
        while True:
            msg = self.recv_json(timeout_ms=200)
            if msg is not None:
                callback(msg)
                count += 1
                if max_messages and count >= max_messages:
                    return
            else:
                time.sleep(idle_sleep)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=100)
        if self._stub is not None:
            self._stub.close()


class Requester:
    """Клиентская сторона req/rep. SYNChronous request → reply."""

    def __init__(
        self,
        endpoint: str,
        *,
        context: Any = None,
        timeout_ms: int = 5000,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self._sock: Any = None
        self._ctx: Any = None
        self._stub: Optional[_StubSocket] = None
        self._init_socket(context)

    def _init_socket(self, context: Any) -> None:
        if not _HAS_ZMQ:
            self._stub = _StubSocket("REQ", self.endpoint)
            return
        self._ctx = context or zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        # `*` используется в bind; для connect нужен конкретный адрес.
        connect_ep = self.endpoint.replace("*", "localhost")
        self._sock.connect(connect_ep)

    def request(self, payload: Any) -> dict[str, Any]:
        """Отправляет запрос и ждёт ответ (с таймаутом).

        Returns:
            Распакованный конверт ответа, либо словарь с ключом ``_error``.
        """
        if self._stub is not None:
            self._stub.send(_wrap("request", payload))
            return {"topic": "reply", "ts_ms": int(time.time() * 1000),
                    "payload": {"ok": False, "reason": "stub_mode"}}
        assert self._sock is not None
        try:
            self._sock.send(_wrap("request", payload))
            raw = self._sock.recv()
        except zmq.Again:
            return {"topic": "reply", "ts_ms": int(time.time() * 1000),
                    "payload": {"ok": False, "reason": "timeout"},
                    "_error": "Таймаут ожидания ответа от replier"}
        except zmq.ZMQError as exc:
            return {"topic": "reply", "ts_ms": int(time.time() * 1000),
                    "payload": {"ok": False, "reason": f"zmq_error: {exc}"},
                    "_error": f"ZeroMQ-ошибка: {exc}"}
        return _unwrap(raw)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
        if self._stub is not None:
            self._stub.close()


class Replier:
    """Серверная сторона req/rep. Бесконечный цикл приёма запросов."""

    def __init__(
        self,
        endpoint: str,
        handler: Callable[[dict[str, Any]], Any],
        *,
        context: Any = None,
    ) -> None:
        self.endpoint = endpoint
        self.handler = handler
        self._sock: Any = None
        self._ctx: Any = None
        self._stub: Optional[_StubSocket] = None
        self._running = False
        self._init_socket(context)

    def _init_socket(self, context: Any) -> None:
        if not _HAS_ZMQ:
            self._stub = _StubSocket("REP", self.endpoint)
            return
        self._ctx = context or zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(self.endpoint)

    def serve_loop(self, *, max_requests: int = 0, idle_sleep: float = 0.02) -> None:
        """Бесконечный цикл обслуживания запросов.

        Args:
            max_requests: Если > 0 — остановится после N запросов.
            idle_sleep: Пауза когда нет входящих запросов.
        """
        if self._stub is not None:
            console.print(
                "[yellow][stub-ZMQ] Replier.serve_loop() — "
                "режим без pyzmq, цикл не запускается.[/yellow]"
            )
            return
        assert self._sock is not None
        self._running = True
        count = 0
        while self._running:
            if not _has_event(self._sock, 200):
                time.sleep(idle_sleep)
                continue
            try:
                raw = self._sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            env = _unwrap(raw)
            try:
                reply = self.handler(env)
            except Exception as exc:  # noqa: BLE001
                reply = {"ok": False, "reason": f"handler_error: {exc}"}
            self._sock.send(_wrap("reply", reply))
            count += 1
            if max_requests and count >= max_requests:
                break

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self.stop()
        if self._sock is not None:
            self._sock.close(linger=0)
        if self._stub is not None:
            self._stub.close()


# --- Вспомогательные ---
def _has_event(sock: Any, timeout_ms: int) -> bool:
    """Poll-проверка наличия входящего сообщения без блокировки."""
    if not _HAS_ZMQ:
        return False
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    events = dict(poller.poll(timeout=timeout_ms))
    return sock in events


def is_zmq_available() -> bool:
    """Возвращает True, если pyzmq установлен."""
    return _HAS_ZMQ


# --- Демо ---
if __name__ == "__main__":
    # Простой smoke-test: Publisher + Subscriber в одном процессе
    # (в реальности они должны быть в разных процессах)
    console.print(f"[cyan]pyzmq available:[/cyan] {_HAS_ZMQ}")
    if not _HAS_ZMQ:
        console.print(
            "[yellow]Установите pyzmq для полной функциональности: "
            "pip install pyzmq[/yellow]"
        )

    ep = "tcp://127.0.0.1:5599"
    pub = Publisher(ep, TOPIC_CV_DETECTION)
    console.print("[green]Publisher создан[/green]")
    pub.publish({"class_name": "cup", "confidence": 0.91, "x": 0.18})
    pub.close()
    console.print("[green]Publisher закрыт[/green]")
