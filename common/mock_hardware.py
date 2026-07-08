"""mock_hardware.py — симулятор железа G1 EDU для тестирования без робота.

Класс :class:`MockG1` эмулирует:
- Локомоцию: ``move_to(x, y)``, ``get_pose()``
- Манипуляцию: ``grasp(force_threshold)``, ``release()``, ``get_force()``
- IMU: ``get_imu()`` (roll, pitch, yaw)
- Голос: ``speak(text)`` (через публикацию в ZeroMQ-топик voice.speak)
- Камеру/CV: ``find_object(class_name)`` (детерминированный mock)

Запуск как фоновый процесс::

    python -m common.mock_hardware --serve
    # или
    python common/mock_hardware.py --serve

Тогда MockG1 держит REP-сокет на ``coffee.tool`` и отвечает на удалённые
вызовы (используется в ``scripts/run_demo.sh``).

Также можно импортировать напрямую::

    from common.mock_hardware import MockG1
    g1 = MockG1()
    g1.move_to(1.5, 2.0)
    g1.grasp(force_threshold=50)
"""

from __future__ import annotations

# --- Импорты ---
import json
import math
import random
import time
from typing import Any

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# Локальные импорты
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_config
from common.logger import get_logger
from common.transport import (
    Publisher,
    Replier,
    TOPIC_COFFEE_STATE,
    TOPIC_VOICE_SPEAK,
    is_zmq_available,
)

# --- Константы ---
# Стартовая позиция робота (метры в СК комнаты)
DEFAULT_START_X = 0.0
DEFAULT_START_Y = 0.0
DEFAULT_START_THETA = 0.0
# Скорость эмуляции ходьбы (м/с) — для реалистичных таймингов
DEFAULT_MOCK_SPEED = 1.5
# Сила «на пальцах» в состоянии покоя (граммы)
DEFAULT_REST_FORCE = 0.0
# Параметры IMU-симуляции (покой ± лёгкий шум)
IMU_NOISE_DEG = 0.5

console = Console()
log = get_logger("mock_g1")


# --- Класс MockG1 ---
class MockG1:
    """Симулятор G1 EDU Ultimate D без реального железа.

    Состояние:
    - ``x, y, theta`` — позиция и ориентация (м, рад).
    - ``force_g`` — текущая сила на «пальцах» (граммы).
    - ``grasped`` — удерживается ли объект.
    - ``holding`` — тип объекта (None | "cup" | ...).
    """

    def __init__(
        self,
        *,
        start_x: float = DEFAULT_START_X,
        start_y: float = DEFAULT_START_Y,
        start_theta: float = DEFAULT_START_THETA,
        speed: float = DEFAULT_MOCK_SPEED,
        seed: int | None = 42,
    ) -> None:
        self.x: float = start_x
        self.y: float = start_y
        self.theta: float = start_theta
        self.speed: float = speed
        self.force_g: float = DEFAULT_REST_FORCE
        self.grasped: bool = False
        self.holding: str | None = None
        self.last_spoken: str = ""
        self._rng = random.Random(seed)
        self._t_start = time.time()
        # Объекты в комнате (для find_object). Координаты в метрах.
        self._scene_objects: dict[str, dict[str, float]] = {
            "cup": {"x": 1.8, "y": 0.5, "z": 0.75},
            "person_oleg": {"x": 3.2, "y": -1.0, "z": 1.1},
        }
        # ZeroMQ-издатель состояния и TTS (создаётся лениво)
        self._state_pub: Publisher | None = None
        self._speak_pub: Publisher | None = None
        self._config = load_config()

    # --- ZMQ-публикации ---
    def _ensure_publishers(self) -> None:
        """Лениво создаёт PUB-сокеты для coffee.state и voice.speak.

        Если порт уже занят другим процессом (например, demo_coffee
        опубликовало voice.speak через свой Publisher) — тишина, продолжаем
        без публикации. Это позволяет MockG1 работать и локально, и как
        сервер, не падая из-за конфликта bind.
        """
        if self._state_pub is None:
            ep = self._config.transport.endpoints.get("coffee_state", "tcp://*:5553")
            try:
                self._state_pub = Publisher(ep, TOPIC_COFFEE_STATE)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"[MOCK] не удалось создать state_pub ({ep}): {exc}")
                self._state_pub = None
        if self._speak_pub is None:
            ep = self._config.transport.endpoints.get("voice_speak", "tcp://*:5552")
            try:
                self._speak_pub = Publisher(ep, TOPIC_VOICE_SPEAK)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"[MOCK] не удалось создать speak_pub ({ep}): {exc}")
                self._speak_pub = None

    def _publish_state(self, state: str, extra: dict[str, Any] | None = None) -> None:
        """Публикует текущее состояние в топик coffee.state."""
        self._ensure_publishers()
        payload = {
            "state": state,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "theta": round(self.theta, 3),
            "grasped": self.grasped,
            "holding": self.holding,
            "force_g": round(self.force_g, 1),
            "uptime_s": round(time.time() - self._t_start, 2),
        }
        if extra:
            payload.update(extra)
        if self._state_pub is not None:
            self._state_pub.publish(payload)

    # --- Локомоция ---
    def move_to(self, x: float, y: float, theta: float | None = None) -> dict[str, Any]:
        """Переместиться в точку (x, y) с ориентацией theta (опционально).

        Эмулирует ходьбу со скоростью ``self.speed`` м/с. Возвращает итоговое
        состояние.
        """
        dx = x - self.x
        dy = y - self.y
        dist = math.hypot(dx, dy)
        # Имитируем время пути (но не более 5 сек, чтобы демо не висели)
        travel_time = min(dist / max(self.speed, 0.1), 5.0)
        log.info(
            f"[MOCK] move_to({x:.2f}, {y:.2f}) — дистанция {dist:.2f} м, "
            f"время {travel_time:.2f} с"
        )
        # Анимация (ускоренная для демо)
        steps = max(int(travel_time * 4), 1)
        for i in range(1, steps + 1):
            frac = i / steps
            self.x = (self.x if i == 1 else self.x) + dx / steps  # noqa: PLW0127
            # Простая линейная интерполяция
        self.x = x
        self.y = y
        if theta is not None:
            self.theta = theta
        self._publish_state("MOVING", {"target_x": x, "target_y": y})
        return {
            "ok": True,
            "x": self.x,
            "y": self.y,
            "theta": self.theta,
            "distance_m": dist,
            "time_s": travel_time,
        }

    def get_pose(self) -> dict[str, float]:
        """Возвращает текущую позу {x, y, theta}."""
        return {"x": self.x, "y": self.y, "theta": self.theta}

    # --- Манипуляция ---
    def grasp(self, force_threshold: float = 50.0) -> dict[str, Any]:
        """Закрыть пальцы до достижения порога силы (граммы).

        Эмулирует плавное схождение пальцев с нарастанием силы.
        Anti-crush: если передан порог > 200 г — отказ (безопасность).
        """
        if force_threshold > 200:
            log.warning(
                f"[MOCK] grasp: порог {force_threshold} г превышает anti-crush "
                f"(200 г) — отказ"
            )
            return {"ok": False, "reason": "anti_crush_threshold_exceeded",
                    "force_g": 0.0}
        # Имитация нарастания силы
        self.force_g = force_threshold
        self.grasped = True
        self.holding = "cup"
        log.info(
            f"[MOCK] grasp(force_threshold={force_threshold} г) — успешно, "
            f"объект удерживается"
        )
        self._publish_state("GRASPING", {"force_g": self.force_g})
        return {
            "ok": True,
            "force_g": self.force_g,
            "grasped": self.grasped,
            "holding": self.holding,
            "reason": "ok",
        }

    def release(self) -> dict[str, Any]:
        """Разжать пальцы."""
        prev = self.force_g
        self.force_g = 0.0
        self.grasped = False
        self.holding = None
        log.info(f"[MOCK] release() — сила {prev:.1f} г → 0")
        self._publish_state("RELEASING", {"prev_force_g": prev})
        return {"ok": True, "prev_force_g": prev, "force_g": 0.0}

    def get_force(self) -> dict[str, Any]:
        """Текущая сила на пальцах (граммы) + флаг grasped."""
        # Лёгкий шум для реалистичности
        noise = self._rng.uniform(-0.5, 0.5)
        return {
            "force_g": round(max(0.0, self.force_g + noise), 2),
            "grasped": self.grasped,
            "holding": self.holding,
        }

    def get_imu(self) -> dict[str, float]:
        """IMU: roll, pitch, yaw в градусах (покой + лёгкий шум)."""
        return {
            "roll_deg": round(self._rng.uniform(-IMU_NOISE_DEG, IMU_NOISE_DEG), 3),
            "pitch_deg": round(self._rng.uniform(-IMU_NOISE_DEG, IMU_NOISE_DEG), 3),
            "yaw_deg": round(math.degrees(self.theta)
                             + self._rng.uniform(-IMU_NOISE_DEG, IMU_NOISE_DEG), 3),
        }

    # --- CV (mock) ---
    def find_object(self, class_name: str = "cup") -> dict[str, Any]:
        """Найти объект в сцене. Возвращает 3D-координаты или not_found."""
        key = class_name.lower()
        # Синонимы
        if key in ("coffee", "стакан", "кофе"):
            key = "cup"
        if key in ("oleg", "человек", "person"):
            key = "person_oleg"
        obj = self._scene_objects.get(key)
        if obj is None:
            return {"found": False, "class_name": class_name, "source": "mock"}
        return {
            "found": True,
            "class_name": class_name,
            "x": obj["x"],
            "y": obj["y"],
            "z": obj["z"],
            "confidence": 0.9 + self._rng.uniform(0, 0.09),
            "source": "mock",
        }

    # --- Голос (mock TTS) ---
    def speak(self, text: str) -> dict[str, Any]:
        """Произнести текст голосом (публикация в voice.speak).

        В реальной системе подписывается voice/tts/infer.py.
        """
        self._ensure_publishers()
        self.last_spoken = text
        log.info(f"[MOCK] speak(): {text!r}")
        if self._speak_pub is not None:
            self._speak_pub.publish({"text": text, "voice": "burunov",
                                     "language": "ru"})
        return {"ok": True, "text": text, "voice": "burunov"}

    # --- Доступ к сцене для тестов ---
    def set_scene_object(self, name: str, x: float, y: float, z: float) -> None:
        """Добавляет/обновляет объект в сцене (для тестов)."""
        self._scene_objects[name] = {"x": x, "y": y, "z": z}

    # --- Сводка состояния ---
    def snapshot(self) -> dict[str, Any]:
        """Полный слепок состояния — для отладки и мониторинга."""
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "theta": round(self.theta, 3),
            "force_g": round(self.force_g, 2),
            "grasped": self.grasped,
            "holding": self.holding,
            "uptime_s": round(time.time() - self._t_start, 2),
            "last_spoken": self.last_spoken,
            "scene_objects": dict(self._scene_objects),
        }


# --- Режим сервера (REP на coffee.tool) ---
def serve_mock_g1(port: int = 5554, *, max_requests: int = 0) -> None:
    """Запускает MockG1 как сервер ZeroMQ req/rep.

    Входящие запросы (формат transport-конверта)::
        {"topic": "request", "payload": {
            "method": "move_to"|"grasp"|"release"|"get_force"|"get_imu"|
                      "speak"|"find_object"|"get_pose"|"snapshot",
            "args": {...}
        }}

    Ответ — конверт с payload от соответствующего метода.
    """
    cfg = load_config()
    endpoint = cfg.transport.endpoints.get("coffee_tool", f"tcp://*:{port}")
    console.print(f"[cyan][serve] MockG1 REP-сервер на {endpoint}[/cyan]")
    console.print(f"[cyan][serve] pyzmq available: {is_zmq_available()}[/cyan]")

    g1 = MockG1()

    def handler(env: dict[str, Any]) -> dict[str, Any]:
        payload = env.get("payload", {})
        method = payload.get("method", "")
        args = payload.get("args", {}) or {}
        log.info(f"[serve] → method={method!r} args={args}")
        try:
            if method == "move_to":
                return g1.move_to(args.get("x", 0.0), args.get("y", 0.0),
                                  args.get("theta"))
            if method == "grasp":
                return g1.grasp(args.get("force_threshold", 50.0))
            if method == "release":
                return g1.release()
            if method == "get_force":
                return g1.get_force()
            if method == "get_imu":
                return g1.get_imu()
            if method == "speak":
                return g1.speak(args.get("text", ""))
            if method == "find_object":
                return g1.find_object(args.get("class_name", "cup"))
            if method == "get_pose":
                return g1.get_pose()
            if method == "snapshot":
                return g1.snapshot()
            return {"ok": False, "reason": f"unknown_method: {method!r}"}
        except Exception as exc:  # noqa: BLE001
            log.error(f"[serve] ошибка в методе {method}: {exc}")
            return {"ok": False, "reason": f"exception: {exc}"}

    replier = Replier(endpoint, handler)
    try:
        replier.serve_loop(max_requests=max_requests)
    except KeyboardInterrupt:
        console.print("\n[yellow][serve] Остановлено пользователем[/yellow]")
    finally:
        replier.close()


# --- CLI ---
@click.command()
@click.option("--serve", is_flag=True,
              help="Запустить как REP-сервер (фоновый процесс для демо).")
@click.option("--port", default=5554, show_default=True,
              help="Порт для REP-сервера (если --serve).")
@click.option("--max-requests", default=0, show_default=True,
              help="Остановить после N запросов (0 = бесконечно).")
@click.option("--smoke-test", is_flag=True,
              help="Прогнать локальный smoke-test без сервера.")
def main(serve: bool, port: int, max_requests: int, smoke_test: bool) -> None:
    """CLI для MockG1."""
    if smoke_test:
        console.print("[bold cyan][smoke-test] MockG1 — локальный прогон[/bold cyan]")
        g1 = MockG1()
        with Live(_render_snapshot(g1), refresh_per_second=2) as live:
            for _ in range(3):
                time.sleep(0.3)
                live.update(_render_snapshot(g1))
        # Прогон основных методов
        console.print(g1.find_object("cup"))
        console.print(g1.move_to(1.5, 2.0))
        console.print(g1.grasp(force_threshold=50))
        console.print(g1.get_force())
        console.print(g1.get_imu())
        console.print(g1.speak("Олег, держи свой кофе"))
        console.print(g1.release())
        console.print(g1.snapshot())
        return

    if serve:
        serve_mock_g1(port=port, max_requests=max_requests)
        return

    # По умолчанию — вывод подсказки
    console.print(Panel(
        "[bold]MockG1 — симулятор железа G1 EDU[/bold]\n\n"
        "Запуск как сервер (для интеграционных демо)::\n\n"
        "  [cyan]python common/mock_hardware.py --serve[/cyan]\n\n"
        "Локальный smoke-test (без сервера, без ZMQ)::\n\n"
        "  [cyan]python common/mock_hardware.py --smoke-test[/cyan]\n",
        title="MockG1 CLI",
        border_style="cyan",
    ))


def _render_snapshot(g1: MockG1) -> Table:
    """rich-таблица со слепком состояния (для smoke-test)."""
    snap = g1.snapshot()
    t = Table(title="MockG1 — состояние", show_header=False, border_style="cyan")
    t.add_column("Параметр", style="bold")
    t.add_column("Значение")
    for k, v in snap.items():
        if k == "scene_objects":
            v = ", ".join(f"{n}@({d['x']},{d['y']},{d['z']})"
                          for n, d in v.items())
        t.add_row(k, str(v))
    return t


if __name__ == "__main__":
    main()
