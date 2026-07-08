"""demo_coffee.py — end-to-end демо: «принеси кофе Олегу».

Сценарий:
1. Голосовая команда «принеси кофе Олегу» (mock ASR → VoiceCommand).
2. Оркестратор разбирает интент (mock parser → intent="coffee").
3. CV находит стакан в сцене (mock → 3D-координаты).
4. Локомоция подходит к стакану (mock G1.move_to).
5. Манипуляция берёт стакан через force feedback (mock G1.grasp).
6. Stability check (mock G1.get_force — стабильно).
7. Локомоция идёт к Олегу (mock G1.move_to).
8. Handover: робот поднимает руку, ждёт приёма (mock G1 + force<20 г).
9. Голосовое подтверждение (mock TTS через G1.speak).

Все шаги логируются через rich с цветными плашками. Запускается без
железа и без vLLM/Qdrant — по умолчанию всё mock.

Запуск::

    python integration/demo_coffee.py
    python integration/demo_coffee.py --command "принеси кофе Олегу"
    python integration/demo_coffee.py --use-zmq   # через MockG1-сервер
"""

from __future__ import annotations

# --- Импорты ---
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Локальные импорты (добавляем корень проекта в sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_config
from common.logger import get_logger
from common.mock_hardware import MockG1
from common.state import VoiceCommand, Detection3D, FaceMatch, GraspResult, NavigationGoal, to_json
from common.transport import (
    Publisher,
    Requester,
    TOPIC_VOICE_COMMAND,
    TOPIC_COFFEE_STATE,
    TOPIC_VOICE_SPEAK,
    TOPIC_CV_DETECTION,
    TOPIC_CV_FACE,
    is_zmq_available,
)

# --- Константы ---
DEFAULT_COMMAND = "принеси кофе Олегу"
DEFAULT_TARGET_PERSON = "Олег"

console = Console()
log = get_logger("demo_coffee")


# --- Заглушки модулей ---
def mock_asr(command: str) -> VoiceCommand:
    """Mock ASR: текст уже распознан, возвращаем VoiceCommand.

    В реальной системе — Whisper на микрофоне G1.
    """
    log.info(f"[ASR] распознан текст: {command!r}")
    cmd = command.lower()
    intent = "unknown"
    params: dict[str, Any] = {}
    if "кофе" in cmd or "coffee" in cmd:
        intent = "coffee"
        params["item"] = "coffee"
    if "олег" in cmd or "oleg" in cmd:
        params["target_person"] = "Oleg"
    if "анекдот" in cmd or "joke" in cmd:
        intent = "joke"
    return VoiceCommand(
        text=command,
        intent=intent,
        params=params,
        confidence=0.93,
        source="mock",
        timestamp_ms=int(time.time() * 1000),
    )


def mock_cv_find_cup(g1: MockG1) -> Detection3D:
    """Mock CV: находит стакан через G1.find_object.

    В реальной системе — coffee/detection/infer.py (YOLO + RealSense depth).
    """
    raw = g1.find_object("cup")
    if not raw.get("found"):
        raise RuntimeError("Стакан не найден в сцене")
    det = Detection3D(
        class_name="cup",
        confidence=raw.get("confidence", 0.9),
        x=raw["x"],
        y=raw["y"],
        z=raw["z"],
        bbox_xyxy=[320, 240, 380, 360],
        source="mock",
        timestamp_ms=int(time.time() * 1000),
    )
    log.info(f"[CV] Detection3D: {to_json(det)}")
    return det


def mock_cv_find_person(g1: MockG1, name: str = "Oleg") -> FaceMatch:
    """Mock CV: находит лицо человека через G1.find_object.

    В реальной системе — coffee/detection/face_id.py (face_recognition).
    """
    raw = g1.find_object(name)
    face = FaceMatch(
        name=name,
        distance=0.42,
        matched=True,
        x=raw.get("x", 0.0),
        y=raw.get("y", 0.0),
        z=raw.get("z", 0.0),
        confidence=0.88,
        timestamp_ms=int(time.time() * 1000),
    )
    log.info(f"[CV] FaceMatch: {to_json(face)}")
    return face


# --- Шаги демо ---
def step_publish_voice_command(pub: Publisher, cmd: VoiceCommand) -> None:
    """Шаг 1: публикуем голосовую команду в шину."""
    console.print(Panel(
        f"[bold magenta]Шаг 1: ASR → voice.command[/bold magenta]\n"
        f"Текст: {cmd.text}\n"
        f"Интент: {cmd.intent}\n"
        f"Параметры: {json.dumps(cmd.params, ensure_ascii=False)}",
        border_style="magenta",
    ))
    pub.publish({
        "text": cmd.text,
        "intent": cmd.intent,
        "params": cmd.params,
        "confidence": cmd.confidence,
        "source": cmd.source,
    })
    time.sleep(0.3)


def step_find_cup(g1: MockG1) -> Detection3D:
    """Шаг 3: CV находит стакан."""
    console.print(Panel("[bold green]Шаг 3: CV → cv.detection (стакан)[/bold green]",
                        border_style="green"))
    det = mock_cv_find_cup(g1)
    console.print(f"  → координаты стакана: ({det.x:.2f}, {det.y:.2f}, {det.z:.2f}) м, "
                  f"conf={det.confidence:.2f}")
    return det


def step_approach_cup(g1: MockG1, det: Detection3D, stop_dist: float) -> None:
    """Шаг 4: локомоция подходит к стакану."""
    console.print(Panel(
        f"[bold yellow]Шаг 4: Locomotion → подход к стакану[/bold yellow]\n"
        f"Цель: ({det.x:.2f}, {det.y:.2f}), остановка в {stop_dist:.1f} м",
        border_style="yellow",
    ))
    # Упрощённо — подходим к точке стакана с отступом
    tx = det.x - stop_dist
    ty = det.y
    result = g1.move_to(tx, ty)
    console.print(f"  → достиг ({result['x']:.2f}, {result['y']:.2f}), "
                  f"дистанция {result['distance_m']:.2f} м за {result['time_s']:.2f} с")


def step_grasp(g1: MockG1, force_thr: int) -> GraspResult:
    """Шаг 5: манипуляция берёт стакан."""
    console.print(Panel(
        f"[bold cyan]Шаг 5: Manipulation → хват стакана (force threshold={force_thr} г)[/bold cyan]",
        border_style="cyan",
    ))
    result = g1.grasp(force_threshold=float(force_thr))
    if not result.get("ok"):
        raise RuntimeError(f"Хват не удался: {result.get('reason')}")
    res = GraspResult(
        success=True,
        force_g=result["force_g"],
        stable=True,
        reason="ok",
        attempts=1,
        timestamp_ms=int(time.time() * 1000),
    )
    console.print(f"  → сила на пальцах: {res.force_g:.1f} г, "
                  f"удерживается: {g1.holding!r}")
    return res


def step_stability_check(g1: MockG1) -> bool:
    """Шаг 6: проверка стабильности хвата."""
    console.print(Panel("[bold cyan]Шаг 6: Stability check[/bold cyan]",
                        border_style="cyan"))
    force_before = g1.get_force()["force_g"]
    time.sleep(0.3)
    force_after = g1.get_force()["force_g"]
    delta_pct = abs(force_after - force_before) / max(force_before, 1.0) * 100
    stable = delta_pct < 20.0
    console.print(f"  → сила до/после: {force_before:.1f}/{force_after:.1f} г, "
                  f"Δ={delta_pct:.1f}% → {'стабильно' if stable else 'НЕ стабильно'}")
    if not stable:
        raise RuntimeError("Stability check не пройден — проскальзывание")
    return True


def step_find_person(g1: MockG1, name: str) -> FaceMatch:
    """Шаг 7: CV находит Олега."""
    console.print(Panel(f"[bold green]Шаг 7: CV → cv.face (поиск {name})[/bold green]",
                        border_style="green"))
    face = mock_cv_find_person(g1, name)
    console.print(f"  → лицо найдено в ({face.x:.2f}, {face.y:.2f}, {face.z:.2f}), "
                  f"distance={face.distance:.2f}")
    return face


def step_approach_target(g1: MockG1, face: FaceMatch, stop_dist: float) -> None:
    """Шаг 8: локомоция подходит к Олегу."""
    console.print(Panel(
        f"[bold yellow]Шаг 8: Locomotion → подход к Олегу[/bold yellow]\n"
        f"Цель: ({face.x:.2f}, {face.y:.2f}), остановка в {stop_dist:.1f} м",
        border_style="yellow",
    ))
    tx = face.x - stop_dist
    ty = face.y
    result = g1.move_to(tx, ty)
    console.print(f"  → достиг ({result['x']:.2f}, {result['y']:.2f}) за {result['time_s']:.2f} с")


def step_handover(g1: MockG1, release_force: int, timeout_s: float = 5.0) -> None:
    """Шаг 9: handover — ждём приём стакана (force < release_force)."""
    console.print(Panel(
        f"[bold cyan]Шаг 9: Handover → ожидание приёма (force < {release_force} г)[/bold cyan]",
        border_style="cyan",
    ))
    start = time.time()
    # Эмулируем приём через 1.5 сек
    accept_after = 1.5
    while time.time() - start < timeout_s:
        elapsed = time.time() - start
        if elapsed >= accept_after:
            # Олег взял стакан — сила падает
            g1.force_g = float(release_force - 5)
            g1.grasped = False
            g1.holding = None
            console.print(f"  → Олег принял стакан (force={g1.force_g:.1f} г < {release_force} г)")
            return
        time.sleep(0.2)
    raise RuntimeError(f"Таймаут handover — Олег не принял стакан за {timeout_s} с")


def step_speak(g1: MockG1, text: str) -> None:
    """Шаг 10: голосовое подтверждение (mock TTS)."""
    console.print(Panel(
        f"[bold magenta]Шаг 10: TTS (Бурунов) → voice.speak[/bold magenta]\n"
        f"Текст: {text!r}",
        border_style="magenta",
    ))
    g1.speak(text)


# --- Main pipeline ---
def run_demo(
    command: str,
    *,
    use_zmq: bool = False,
    target_person: str = DEFAULT_TARGET_PERSON,
) -> dict[str, Any]:
    """Прогоняет end-to-end демо. Возвращает отчёт."""
    cfg = load_config()
    console.print(Panel(
        f"[bold]G1 EDU — demo_coffee[/bold]\n"
        f"Команда: {command!r}\n"
        f"Цель: {target_person}\n"
        f"ZMQ: {is_zmq_available() and use_zmq}\n"
        f"pyzmq available: {is_zmq_available()}",
        title="Запуск demo_coffee",
        border_style="bright_blue",
    ))

    # Издатели (создаются лениво).
    # vc_pub — публикует голосовую команду от ASR (это задача демо).
    # state_pub и speak_pub в локальном режиме создаёт сам MockG1
    # (bindит tcp://*:5553 и tcp://*:5552). В удалённом режиме speak_pub
    # нужен для _RemoteG1.speak (он только публикует, не bindит — но bindит
    # demo_coffee, т.к. именно demo является источником voice.speak в этом случае).
    vc_pub = Publisher(
        cfg.transport.endpoints.get("voice_command", "tcp://*:5551"),
        TOPIC_VOICE_COMMAND,
    )
    state_pub: Publisher | None = None
    speak_pub: Publisher | None = None

    # Если use_zmq — подключаемся к MockG1-серверу через Requester
    g1: Any
    requester: Requester | None = None
    if use_zmq and is_zmq_available():
        # В удалённом режиме speak_pub создаём — _RemoteG1.speak будет публиковать
        speak_pub = Publisher(
            cfg.transport.endpoints.get("voice_speak", "tcp://*:5552"),
            TOPIC_VOICE_SPEAK,
        )
        requester = Requester(
            cfg.transport.endpoints.get("coffee_tool", "tcp://localhost:5554"),
        )

        class _RemoteG1:
            """Thin-обёртка, вызывает MockG1-сервер через ZMQ req/rep."""

            def __init__(self, rq: Requester, sp: Publisher) -> None:
                self._rq = rq
                self._sp = sp
                self.holding: str | None = None
                self.grasped = False
                self.force_g = 0.0
                self.x = 0.0
                self.y = 0.0
                self.theta = 0.0

            def _call(self, method: str, **args: Any) -> dict[str, Any]:
                env = self._rq.request({"method": method, "args": args})
                return env.get("payload", {})

            def move_to(self, x: float, y: float, theta: float | None = None) -> dict[str, Any]:
                a = {"x": x, "y": y}
                if theta is not None:
                    a["theta"] = theta
                r = self._call("move_to", **a)
                if r.get("ok"):
                    self.x = r.get("x", x)
                    self.y = r.get("y", y)
                return r

            def grasp(self, force_threshold: float = 50.0) -> dict[str, Any]:
                r = self._call("grasp", force_threshold=force_threshold)
                if r.get("ok"):
                    self.grasped = True
                    self.holding = "cup"
                    self.force_g = r.get("force_g", force_threshold)
                return r

            def release(self) -> dict[str, Any]:
                r = self._call("release")
                self.grasped = False
                self.holding = None
                self.force_g = 0.0
                return r

            def get_force(self) -> dict[str, Any]:
                r = self._call("get_force")
                if r.get("grasped") is not None:
                    self.grasped = r["grasped"]
                    self.holding = r.get("holding")
                return r

            def get_imu(self) -> dict[str, Any]:
                return self._call("get_imu")

            def speak(self, text: str) -> dict[str, Any]:
                self._sp.publish({"text": text, "voice": "burunov", "language": "ru"})
                return {"ok": True, "text": text}

            def find_object(self, class_name: str = "cup") -> dict[str, Any]:
                return self._call("find_object", class_name=class_name)

            def snapshot(self) -> dict[str, Any]:
                return self._call("snapshot")

        g1 = _RemoteG1(requester, speak_pub)
        log.info("[demo_coffee] режим: удалённый MockG1 через ZMQ (speak_pub создан)")
    else:
        g1 = MockG1()
        # Локальный MockG1 сам создаёт speak_pub и state_pub (lazy) и bindит порты
        log.info("[demo_coffee] режим: локальный MockG1 (без ZMQ-сервера)")

    report: dict[str, Any] = {"steps": [], "errors": [], "duration_s": 0.0}
    t_start = time.time()

    def _step(name: str, fn: Callable[[], Any]) -> Any:
        try:
            console.print()
            if state_pub is not None:
                state_pub.publish({"state": name})
            result = fn()
            report["steps"].append({"name": name, "ok": True})
            return result
        except Exception as exc:  # noqa: BLE001
            report["errors"].append({"step": name, "error": str(exc)})
            report["steps"].append({"name": name, "ok": False, "error": str(exc)})
            log.error(f"[demo_coffee] шаг {name!r} провален: {exc}")
            raise

    try:
        # 1. ASR → voice.command
        cmd = mock_asr(command)
        step_publish_voice_command(vc_pub, cmd)

        # 2. Оркестратор: разбор интента (mock)
        console.print(Panel(
            f"[bold blue]Шаг 2: Orchestrator → разбор интента[/bold blue]\n"
            f"intent={cmd.intent}, params={json.dumps(cmd.params, ensure_ascii=False)}",
            border_style="blue",
        ))
        if cmd.intent != "coffee":
            raise RuntimeError(f"Неожиданный интент: {cmd.intent!r} (ожидался 'coffee')")
        target_person_norm = cmd.params.get("target_person", target_person)

        # 3. CV находит стакан
        det = _step("FIND_CUP", lambda: step_find_cup(g1))

        # 4. Локомоция подходит к стакану
        _step("APPROACH_CUP",
              lambda: step_approach_cup(g1, det, cfg.robot.approach_distance))

        # 5. Хват
        _step("GRASP",
              lambda: step_grasp(g1, cfg.robot.grasp_force_min))

        # 6. Stability check
        _step("STABILITY_CHECK", lambda: step_stability_check(g1))

        # 7. CV находит Олега
        face = _step("FIND_TARGET",
                     lambda: step_find_person(g1, target_person_norm))

        # 8. Локомоция к Олегу
        _step("APPROACH_TARGET",
              lambda: step_approach_target(g1, face, cfg.robot.handover_distance))

        # 9. Handover
        _step("HANDOVER",
              lambda: step_handover(g1, cfg.robot.handover_release_force))

        # 10. Голосовое подтверждение
        _step("SPEAK",
              lambda: step_speak(g1, f"{target_person}, держи свой кофе"))

    except Exception as exc:  # noqa: BLE001
        console.print(f"\n[red]Демо прервано: {exc}[/red]")
        # Голосовое сообщение об ошибке
        try:
            g1.speak("Произошла ошибка. Кофе не доставлен.")
        except Exception:
            pass
    finally:
        report["duration_s"] = round(time.time() - t_start, 2)
        vc_pub.close()
        if state_pub is not None:
            state_pub.close()
        if speak_pub is not None:
            speak_pub.close()
        if requester is not None:
            requester.close()

    # Итоговый отчёт
    _print_report(report)
    return report


def _print_report(report: dict[str, Any]) -> None:
    """Выводит rich-таблицу с итогами демо."""
    t = Table(title="demo_coffee — отчёт", border_style="bright_blue",
              show_lines=True)
    t.add_column("Шаг", style="bold")
    t.add_column("OK", justify="center")
    t.add_column("Комментарий")
    for s in report["steps"]:
        ok = "✓" if s["ok"] else "✗"
        color = "green" if s["ok"] else "red"
        comment = s.get("error", "")
        t.add_row(s["name"], f"[{color}]{ok}[/{color}]", comment)
    console.print(t)
    console.print(f"[bold]Длительность:[/bold] {report['duration_s']} с")
    if report["errors"]:
        console.print(f"[red]Ошибок:[/red] {len(report['errors'])}")
    else:
        console.print("[green]Все шаги выполнены успешно.[/green]")


# --- CLI ---
@click.command()
@click.option("--command", default=DEFAULT_COMMAND, show_default=True,
              help="Голосовая команда для разбора.")
@click.option("--use-zmq", is_flag=True,
              help="Использовать удалённый MockG1 через ZeroMQ (запустите "
                   "MockG1 --serve в фоне).")
@click.option("--target-person", default=DEFAULT_TARGET_PERSON, show_default=True,
              help="Имя целевого человека (для CV face_id).")
def main(command: str, use_zmq: bool, target_person: str) -> None:
    """End-to-end демо «принеси кофе Олегу»."""
    run_demo(command, use_zmq=use_zmq, target_person=target_person)


if __name__ == "__main__":
    main()
