"""demo_joke.py — end-to-end демо: «расскажи анекдот 86-го».

Сценарий:
1. Голосовая команда «расскажи анекдот 86-го» (mock ASR → VoiceCommand).
2. Извлечение интента: intent="joke", topic, year.
3. RAG retrieve:
   - Если Qdrant + sentence-transformers доступны — реальный поиск через
     voice/rag/retrieve.py (субпроцесс, --json).
   - Иначе — лёгкий TF-IDF поиск через voice/rag/retrieve_light.py
     (субпроцесс, --json) по локальному корпусу 150+ анекдотов.
4. TTS: если есть checkpoint Бурунова — реальный синтез через voice/tts/infer.py,
   иначе mock TTS (печать текста + публикация в voice.speak).

Запуск::

    python integration/demo_joke.py
    python integration/demo_joke.py --query "расскажи анекдот про штирлица"
    python integration/demo_joke.py --use-qdrant   # реальный RAG (Qdrant + ST)
    python integration/demo_joke.py --use-tts      # реальный TTS (если есть ckpt)
"""

from __future__ import annotations

# --- Импорты ---
import json
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Локальные импорты
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_config
from common.logger import get_logger
from common.state import VoiceCommand, to_json
from common.transport import (
    Publisher,
    TOPIC_VOICE_COMMAND,
    TOPIC_VOICE_SPEAK,
    is_zmq_available,
)

# --- Константы ---
DEFAULT_QUERY = "расскажи анекдот 86-го"
JOKES_CORPUS_PATH = Path(__file__).resolve().parent.parent / "voice" / "data" / "jokes_corpus.jsonl"
RETRIEVE_SCRIPT = Path(__file__).resolve().parent.parent / "voice" / "rag" / "retrieve.py"
RETRIEVE_LIGHT_SCRIPT = Path(__file__).resolve().parent.parent / "voice" / "rag" / "retrieve_light.py"
TTS_INFER_SCRIPT = Path(__file__).resolve().parent.parent / "voice" / "tts" / "infer.py"

QDRANT_HEALTH_TIMEOUT_S = 1.5

console = Console()
log = get_logger("demo_joke")


# --- Утилиты ---
def mock_asr(query: str) -> VoiceCommand:
    """Mock ASR: распознаёт текст и извлекает интент (joke)."""
    log.info(f"[ASR] распознан текст: {query!r}")
    q = query.lower()
    intent = "joke" if ("анекдот" in q or "joke" in q) else "unknown"
    topic = ""
    # Наивная эвристика темы
    for kw, name in [
        ("штирлиц", "штирлиц"),
        ("чукча", "чукча"),
        ("василий", "василий иванович"),
        ("петка", "василий иванович"),
        ("брежнев", "брежнев"),
        ("поручик", "поручик ржевский"),
        ("вовочка", "школа"),
        ("учитель", "школа"),
        ("участков", "участковый"),
        ("евре", "старые евреи"),
        ("магазин", "магазин"),
        ("доктор", "лечащий врач"),
        ("врач", "лечащий врач"),
    ]:
        if kw in q:
            topic = name
            break
    year = 1986 if "86" in q or "1986" in q else None
    return VoiceCommand(
        text=query,
        intent=intent,
        params={"topic": topic, "year": year},
        confidence=0.9,
        source="mock",
        timestamp_ms=int(time.time() * 1000),
    )


def _qdrant_available(url: str) -> bool:
    """Проверяет доступность Qdrant по HTTP."""
    try:
        with urllib.request.urlopen(f"{url}/healthz", timeout=QDRANT_HEALTH_TIMEOUT_S) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False
    except Exception:  # noqa: BLE001
        return False


def rag_retrieve_via_qdrant(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Реальный RAG-поиск через voice/rag/retrieve.py (subprocess).

    Возвращает список словарей с ключами id, text, topic, year, score.
    """
    if not RETRIEVE_SCRIPT.exists():
        log.warning(f"retrieve.py не найден: {RETRIEVE_SCRIPT}")
        return []
    cmd = [
        sys.executable, str(RETRIEVE_SCRIPT),
        "--query", query,
        "--top-k", str(top_k),
        "--json",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        log.error("retrieve.py таймаут")
        return []
    if proc.returncode != 0:
        log.error(f"retrieve.py упал: {proc.stderr[:300]}")
        return []
    # Парсим stdout — ищем JSON-блок
    out = proc.stdout.strip()
    try:
        # retrieve.py может печатать таблицы + JSON; берём последний {...}
        candidates = [ln.strip() for ln in out.splitlines()
                      if ln.strip().startswith("{") or ln.strip().startswith("[")]
        if candidates:
            # Если последний — массив, берём его; иначе объект с полем results
            data = json.loads(candidates[-1])
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("results", [data])
    except json.JSONDecodeError:
        pass
    return []


def rag_retrieve_via_light(query: str, top_k: int = 3,
                            topic: str | None = None,
                            year: int | None = None) -> list[dict[str, Any]]:
    """Лёгкий RAG-поиск через voice/rag/retrieve_light.py (subprocess, --json).

    TF-IDF на sklearn + nltk Snowball русский стемминг. Не требует
    sentence-transformers / torch / Qdrant.

    Возвращает список словарей с ключами id, text, topic, year, score.
    """
    if not RETRIEVE_LIGHT_SCRIPT.exists():
        log.warning(f"retrieve_light.py не найден: {RETRIEVE_LIGHT_SCRIPT}")
        return []
    cmd = [
        sys.executable, str(RETRIEVE_LIGHT_SCRIPT),
        "--query", query,
        "--top-k", str(top_k),
        "--corpus", str(JOKES_CORPUS_PATH),
        "--json",
    ]
    if topic:
        cmd += ["--topic", topic]
    if year is not None:
        cmd += ["--year", str(year)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        log.error("retrieve_light.py таймаут")
        return []
    if proc.returncode != 0:
        log.error(f"retrieve_light.py упал: {proc.stderr[:300]}")
        return []
    out = proc.stdout.strip()
    try:
        data = json.loads(out)
        if isinstance(data, dict):
            return data.get("results", [])
        if isinstance(data, list):
            return data
    except json.JSONDecodeError as exc:
        log.error(f"retrieve_light.py: не удалось распарсить JSON: {exc}")
    return []


def rag_retrieve_fallback(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Старый наивный fallback: keyword-scoring по корпусу.

    Оставлен как последний рубеж, если даже retrieve_light.py недоступен.
    Считает совпадения ключевых слов запроса в тексте/теме анекдота.
    """
    if not JOKES_CORPUS_PATH.exists():
        log.warning(f"Корпус анекдотов не найден: {JOKES_CORPUS_PATH}")
        return []
    jokes: list[dict[str, Any]] = []
    with JOKES_CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    jokes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not jokes:
        return []

    # Извлекаем ключевые слова из запроса
    q_lower = query.lower()
    keywords = [w for w in q_lower.replace(",", " ").split()
                if len(w) >= 3 and w not in {"анекдот", "про", "про", "год", "года",
                                              "расскажи", "из", "тот", "которые"}]
    scored: list[tuple[float, dict[str, Any]]] = []
    for j in jokes:
        text = (j.get("topic", "") + " " + j.get("text", "")).lower()
        score = 0.0
        for kw in keywords:
            if kw in text:
                score += 1.0
        # Если тема указана в запросе и совпадает — бонус
        if j.get("topic") and j["topic"].lower() in q_lower:
            score += 3.0
        # Если есть год 1986 в запросе — бонус
        if j.get("year") == 1986 and ("86" in q_lower or "1986" in q_lower):
            score += 2.0
        scored.append((score, j))
    scored.sort(key=lambda x: x[0], reverse=True)
    # Если ничего не нашлось по ключевым словам — возвращаем случайные
    if all(s == 0.0 for s, _ in scored[:top_k]):
        rng = random.Random(42)
        picked = rng.sample(jokes, min(top_k, len(jokes)))
        return [{"id": j["id"], "text": j["text"], "topic": j["topic"],
                 "year": j.get("year"), "score": 0.0} for j in picked]
    return [{"id": j["id"], "text": j["text"], "topic": j["topic"],
             "year": j.get("year"), "score": s}
            for s, j in scored[:top_k]]


def mock_tts_speak(pub: Publisher, text: str) -> None:
    """Mock TTS: публикация текста в voice.speak.

    В реальной системе — voice/tts/infer.py с checkpoint Бурунова.
    """
    pub.publish({"text": text, "voice": "burunov", "language": "ru",
                 "sample_rate": 22050, "source": "mock"})
    console.print(f"[magenta][TTS-mock] Бурунов говорит:[/magenta] {text}")


def real_tts_speak(text: str, checkpoint: str, out_path: Path) -> bool:
    """Реальный TTS через voice/tts/infer.py. Возвращает True при успехе."""
    if not TTS_INFER_SCRIPT.exists():
        log.warning(f"voice/tts/infer.py не найден: {TTS_INFER_SCRIPT}")
        return False
    cmd = [
        sys.executable, str(TTS_INFER_SCRIPT),
        "--text", text,
        "--checkpoint", checkpoint,
        "--out", str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        log.error("infer.py таймаут")
        return False
    if proc.returncode != 0:
        log.error(f"infer.py упал (rc={proc.returncode}): {proc.stderr[:300]}")
        return False
    return out_path.exists()


# --- Main pipeline ---
def run_demo(
    query: str,
    *,
    use_qdrant: bool = False,
    use_tts: bool = False,
    top_k: int = 1,
) -> dict[str, Any]:
    """Прогоняет end-to-end демо. Возвращает отчёт."""
    cfg = load_config()
    console.print(Panel(
        f"[bold]G1 EDU — demo_joke[/bold]\n"
        f"Запрос: {query!r}\n"
        f"Qdrant: {cfg.qdrant.url}\n"
        f"TTS checkpoint: {cfg.tts.checkpoint}",
        title="Запуск demo_joke",
        border_style="bright_blue",
    ))

    vc_pub = Publisher(
        cfg.transport.endpoints.get("voice_command", "tcp://*:5551"),
        TOPIC_VOICE_COMMAND,
    )
    speak_pub = Publisher(
        cfg.transport.endpoints.get("voice_speak", "tcp://*:5552"),
        TOPIC_VOICE_SPEAK,
    )

    report: dict[str, Any] = {"steps": [], "jokes": [], "duration_s": 0.0}
    t_start = time.time()

    try:
        # 1. ASR + интент
        console.print(Panel(
            f"[bold magenta]Шаг 1: ASR → voice.command[/bold magenta]\n"
            f"Текст: {query}",
            border_style="magenta",
        ))
        cmd = mock_asr(query)
        vc_pub.publish({
            "text": cmd.text,
            "intent": cmd.intent,
            "params": cmd.params,
            "source": cmd.source,
        })
        console.print(f"  → intent={cmd.intent}, params={json.dumps(cmd.params, ensure_ascii=False)}")
        report["steps"].append({"name": "ASR", "ok": True})

        # 2. RAG retrieve
        console.print(Panel(
            f"[bold green]Шаг 2: RAG retrieve (top_k={top_k})[/bold green]",
            border_style="green",
        ))
        # Принятие решения: 3 уровня приоритета.
        #   1) Qdrant + sentence-transformers (через retrieve.py) — если
        #      пользователь явно просит (--use-qdrant) И Qdrant жив.
        #   2) Лёгкий TF-IDF (через retrieve_light.py) — основной fallback,
        #      не требует torch / sentence-transformers / Qdrant.
        #   3) Наивный keyword-search по корпусу — последний рубеж, если
        #      даже retrieve_light.py упал.
        topic_param = cmd.params.get("topic") or None
        year_param = cmd.params.get("year") or None
        use_real = use_qdrant and _qdrant_available(cfg.qdrant.url)
        if use_qdrant and not use_real:
            console.print(
                f"[yellow]  Qdrant недоступен ({cfg.qdrant.url}) — "
                f"переключаюсь на лёгкий TF-IDF RAG[/yellow]"
            )
        rag_source = "fallback-keyword"
        if use_real:
            console.print(f"  [green]Qdrant доступен — реальный RAG (retrieve.py)[/green]")
            hits = rag_retrieve_via_qdrant(query, top_k=top_k)
            if hits:
                rag_source = "qdrant"
        else:
            console.print(
                f"  [cyan]Лёгкий TF-IDF RAG (retrieve_light.py), "
                f"корпус: {JOKES_CORPUS_PATH.name}[/cyan]"
            )
            hits = rag_retrieve_via_light(
                query, top_k=top_k,
                topic=topic_param, year=year_param,
            )
            if hits:
                rag_source = "tfidf-light"
        # Последний рубеж: keyword fallback.
        if not hits:
            console.print(
                "[yellow]  retrieve_light.py ничего не вернул — "
                "наивный keyword-fallback по корпусу[/yellow]"
            )
            hits = rag_retrieve_fallback(query, top_k=top_k)
            if hits:
                rag_source = "fallback-keyword"

        if not hits:
            raise RuntimeError("Не найдено ни одного анекдота")
        report["jokes"] = hits
        report["steps"].append({"name": "RAG", "ok": True, "source": rag_source})
        # Печатаем найденные анекдоты
        for i, h in enumerate(hits, 1):
            console.print(
                f"  [{i}] [dim]id={h.get('id')} topic={h.get('topic')} "
                f"year={h.get('year')} score={h.get('score', 0):.2f}[/dim]\n"
                f"      {h.get('text')}"
            )

        # 3. TTS (mock или реальный)
        console.print(Panel(
            f"[bold cyan]Шаг 3: TTS (Бурунов) → voice.speak[/bold cyan]",
            border_style="cyan",
        ))
        text_to_speak = hits[0]["text"]
        if use_tts:
            out_wav = Path("integration/out") / f"joke_{int(time.time())}.wav"
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            ok = real_tts_speak(text_to_speak, cfg.tts.checkpoint, out_wav)
            if ok:
                console.print(f"  [green]TTS синтезирован:[/green] {out_wav}")
                report["steps"].append({"name": "TTS", "ok": True,
                                        "wav": str(out_wav)})
            else:
                console.print(
                    "[yellow]  Реальный TTS не сработал — fallback на mock[/yellow]"
                )
                mock_tts_speak(speak_pub, text_to_speak)
                report["steps"].append({"name": "TTS", "ok": True, "source": "mock"})
        else:
            mock_tts_speak(speak_pub, text_to_speak)
            report["steps"].append({"name": "TTS", "ok": True, "source": "mock"})

    except Exception as exc:  # noqa: BLE001
        console.print(f"\n[red]Демо прервано: {exc}[/red]")
        report["steps"].append({"name": "ERROR", "ok": False, "error": str(exc)})
        report["error"] = str(exc)
    finally:
        report["duration_s"] = round(time.time() - t_start, 2)
        vc_pub.close()
        speak_pub.close()

    # Итоговый отчёт
    _print_report(report)
    return report


def _print_report(report: dict[str, Any]) -> None:
    """Выводит rich-таблицу с итогами."""
    t = Table(title="demo_joke — отчёт", border_style="bright_blue",
              show_lines=True)
    t.add_column("Шаг", style="bold")
    t.add_column("OK", justify="center")
    t.add_column("Источник / комментарий")
    for s in report["steps"]:
        ok = "✓" if s["ok"] else "✗"
        color = "green" if s["ok"] else "red"
        comment = s.get("source", "") or s.get("error", "") or s.get("wav", "")
        t.add_row(s["name"], f"[{color}]{ok}[/{color}]", comment)
    console.print(t)
    console.print(f"[bold]Длительность:[/bold] {report['duration_s']} с")
    if report["jokes"]:
        console.print(f"[bold]Найдено анекдотов:[/bold] {len(report['jokes'])}")


# --- CLI ---
@click.command()
@click.option("--query", default=DEFAULT_QUERY, show_default=True,
              help="Голосовой запрос для разбора.")
@click.option("--use-qdrant", is_flag=True,
              help="Использовать реальный Qdrant (иначе fallback по корпусу).")
@click.option("--use-tts", is_flag=True,
              help="Использовать реальный TTS (нужен checkpoint Бурунова).")
@click.option("--top-k", default=1, show_default=True,
              help="Сколько анекдотов вернуть из RAG.")
def main(query: str, use_qdrant: bool, use_tts: bool, top_k: int) -> None:
    """End-to-end демо «расскажи анекдот 86-го»."""
    run_demo(query, use_qdrant=use_qdrant, use_tts=use_tts, top_k=top_k)


if __name__ == "__main__":
    main()
