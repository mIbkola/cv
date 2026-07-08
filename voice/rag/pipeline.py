"""Полный пайплайн голосового модуля: ввод → интент → RAG → TTS.

Сценарий:
    1. Принимает голосовой (wav) или текстовый ввод.
    2. Если голосовой — ASR через Whisper.
    3. LLM (vLLM локально, Qwen2.5-7B-Instruct) извлекает интент: {action, topic, year}.
       Если vLLM недоступен — fallback на простую эвристику по ключевым словам.
    4. RAG retrieve top-5 в Qdrant (через sentence-transformers).
       Если sentence-transformers или Qdrant недоступны — fallback на лёгкий
       TF-IDF поиск через retrieve_light.py (sklearn + nltk Snowball).
    5. Rerank через cross-encoder → топ-1 (только если sentence-transformers).
    6. Опционально LLM-адаптация (например, добавить «А вы слышали...»).
    7. TTS (Бурунов) → wav 22050 Гц моно.
    8. Сохраняет результат в out/.

Запуск:
    # Текстовый ввод
    python pipeline.py --text "расскажи анекдот 86 года про штирлица"

    # Голосовой ввод
    python pipeline.py --audio ./input.wav

    # Без TTS (только поиск + адаптация текста)
    python pipeline.py --text "..." --no-tts
"""

from __future__ import annotations

# --- Импорты ---------------------------------------------------------------
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# --- Константы -------------------------------------------------------------

DEFAULT_QDRANT_URL: str = "http://localhost:6333"
DEFAULT_COLLECTION: str = "jokes_1986"
DEFAULT_VLLM_URL: str = "http://localhost:8000/v1/chat/completions"
DEFAULT_LLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_WHISPER_MODEL: str = "base"
DEFAULT_TTS_CHECKPOINT: str = "../tts/checkpoints/burunov"
DEFAULT_OUT_DIR: str = "./out"

DEFAULT_TOP_K: int = 5
QDRANT_TIMEOUT_SEC: float = 5.0
VLLM_TIMEOUT_SEC: float = 30.0

# Путь к корпусу анекдотов (используется в fallback на retrieve_light.py).
# Разрешаем относительно расположения этого файла, чтобы работало из любой CWD.
DEFAULT_CORPUS_PATH: str = str(
    Path(__file__).resolve().parent.parent / "data" / "jokes_corpus.jsonl"
)

# Список «знакомых» тем для эвристического fallback'а.
KNOWN_TOPICS: tuple[str, ...] = (
    "штирлиц", "василий иванович", "чукча", "лечащий врач",
    "поручик ржевский", "старые евреи", "участковый", "брежнев",
    "школа", "магазин", "новый русский", "сталин", "ленин",
    "коллективизм", "очередь", "дефицит",
)

# Триггеры действий.
JOKE_TRIGGERS: tuple[str, ...] = (
    "анекдот", "шутк", "расскажи", "историю", "байку",
)

console = Console()


# --- Модельки данных -------------------------------------------------------

@dataclass
class Intent:
    """Извлечённый из запроса интент."""
    action: str = "joke"           # пока поддерживаем только "joke"
    topic: Optional[str] = None
    year: Optional[int] = None
    raw_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    """Полный результат работы пайплайна."""
    input_text: str
    asr_used: bool
    intent: Intent
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    selected: Optional[Dict[str, Any]] = None
    adapted_text: Optional[str] = None
    audio_path: Optional[str] = None
    timings: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


# --- Утилиты ---------------------------------------------------------------

def _check_pkg(import_name: str, pip_name: str) -> None:
    """Проверяет наличие Python-пакета, иначе падает с понятной ошибкой."""
    try:
        __import__(import_name)
    except ImportError:
        console.print(f"[bold red]Ошибка:[/] Python-пакет '{pip_name}' не установлен.")
        console.print(f"   Установите: pip install {pip_name}")
        sys.exit(1)


def _try_import(import_name: str) -> bool:
    """Мягко проверяет наличие пакета. Возвращает True/False, не падает."""
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


# --- Шаг 1: ASR ------------------------------------------------------------

def transcribe_audio(audio_path: Path, whisper_model_name: str) -> str:
    """Распознаёт речь в аудиофайле через Whisper, возвращает текст."""
    try:
        import whisper
    except ImportError:
        console.print("[bold red]Ошибка:[/] openai-whisper не установлен.")
        sys.exit(2)

    if not audio_path.exists():
        console.print(f"[bold red]Ошибка:[/] аудиофайл не найден: {audio_path}")
        sys.exit(3)

    console.print(f"Загрузка Whisper '{whisper_model_name}'...")
    model = whisper.load_model(whisper_model_name)
    console.print(f"ASR: распознаём {audio_path.name}...")
    result = model.transcribe(str(audio_path), language="ru", task="transcribe", verbose=False)
    text: str = result.get("text", "").strip()
    text = " ".join(text.split())
    return text


# --- Шаг 2: Извлечение интента через LLM -----------------------------------

INTENT_SYSTEM_PROMPT: str = (
    "Ты — помощник, который извлекает интент из запроса пользователя на русском. "
    "Запрос — это просьба рассказать анекдот. Нужно вернуть строго JSON с полями: "
    '{"action": "joke", "topic": <строка или null>, "year": <целое число или null>}. '
    "action всегда 'joke'. topic — тема анекдота (например, 'штирлиц', 'чукча', "
    "'василий иванович', 'поручик ржевский', 'брежнев'). year — год, упомянутый в "
    "запросе (например, 1986), или null. Не добавляй никаких комментариев, только JSON."
)


def call_vllm(
    user_text: str,
    vllm_url: str,
    model: str,
    system_prompt: str = INTENT_SYSTEM_PROMPT,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> Optional[str]:
    """Вызывает локальный vLLM через OpenAI-совместимый HTTP API.

    Возвращает текст ответа или None, если vLLM недоступен.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        vllm_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=VLLM_TIMEOUT_SEC) as resp:
            body = resp.read().decode("utf-8")
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        console.print(
            f"[yellow]vLLM недоступен по адресу {vllm_url}.[/]\n"
            f"   Причина: {exc}\n"
            f"   Переключаемся на эвристический fallback."
        )
        return None
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[yellow]vLLM вернул ошибку:[/] {exc}\n"
            f"   Переключаемся на эвристический fallback."
        )
        return None

    try:
        obj = json.loads(body)
        # OpenAI-совместимый формат: choices[0].message.content
        return obj["choices"][0]["message"]["content"].strip()
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        console.print(
            f"[yellow]Не удалось разобрать ответ vLLM:[/] {exc}\n"
            f"   Переключаемся на эвристический fallback."
        )
        return None


def parse_intent_json(text: str) -> Optional[Intent]:
    """Парсит JSON интента из ответа LLM. Возвращает None при неудаче."""
    # LLM может обернуть JSON в ```json ... ``` — вырежем.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    action = str(obj.get("action", "joke"))
    topic = obj.get("topic")
    year = obj.get("year")
    if topic in ("", "null", "None"):
        topic = None
    if year in (None, "null", "None", ""):
        year = None
    else:
        try:
            year = int(year)
        except (ValueError, TypeError):
            year = None
    return Intent(action=action, topic=topic, year=year)


def fallback_intent(user_text: str) -> Intent:
    """Эвристический fallback извлечения интента.

    Срабатывает, если vLLM недоступен. Ищет ключевые слова в тексте.
    Учитывает русскую морфологию: тема «чукча» должна матчится на
    «чукчу»/«чукче»/«чукчей» и т.п. — используем nltk Snowball stemmer
    (если доступен), иначе простой substring match.
    """
    text_lower = user_text.lower()

    # Год: ищем 4-значное число 19xx/20xx.
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", text_lower)
    year = int(year_match.group(1)) if year_match else None

    # Тема: пытаемся найти известную тему в тексте.
    topic: Optional[str] = None
    # Сначала — прямой substring (для многословных тем типа «василий иванович’).
    for t in KNOWN_TOPICS:
        if t in text_lower:
            topic = t
            break
    # Если прямой матч не сработал — пробуем стемминг (для однословных тем).
    if topic is None:
        try:
            from nltk.stem.snowball import RussianStemmer
            stemmer = RussianStemmer()
            text_stems = {stemmer.stem(w) for w in re.findall(r"[а-яё]+", text_lower)}
            for t in KNOWN_TOPICS:
                # Для многословных тем берём стем первого слова.
                t_stem = stemmer.stem(t.split()[0])
                if t_stem in text_stems:
                    topic = t
                    break
        except ImportError:
            pass  # без nltk — оставляем None

    # Action: всегда 'joke' (pipeline пока не поддерживает другие).
    action = "joke"
    if not any(tr in text_lower for tr in JOKE_TRIGGERS):
        # Если триггеров нет — считаем, что всё равно шутка (по умолчанию).
        action = "joke"

    return Intent(action=action, topic=topic, year=year, raw_query=user_text)


def extract_intent(
    user_text: str,
    vllm_url: str,
    llm_model: str,
) -> Intent:
    """Извлекает интент: пытается через vLLM, иначе fallback."""
    console.print("Извлечение интента через LLM (vLLM)...")
    t0 = time.time()
    reply = call_vllm(user_text, vllm_url=vllm_url, model=llm_model)
    dt = time.time() - t0
    if reply is None:
        intent = fallback_intent(user_text)
        intent.raw_query = user_text
        console.print(
            f"[yellow]Использован эвристический fallback ({dt:.2f} сек):[/]\n"
            f"   {intent.to_dict()}"
        )
        return intent

    intent = parse_intent_json(reply)
    if intent is None:
        console.print(
            f"[yellow]LLM вернул невалидный JSON, fallback на эвристику.[/]\n"
            f"   Ответ LLM: {reply[:200]}"
        )
        intent = fallback_intent(user_text)
    intent.raw_query = user_text
    console.print(
        f"[green]LLM ({dt:.2f} сек):[/]\n"
        f"   {intent.to_dict()}"
    )
    return intent


# --- Шаг 3: RAG retrieve ---------------------------------------------------

def retrieve_jokes(
    intent: Intent,
    qdrant_url: str,
    collection: str,
    embedder,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Ищет top-k анекдотов в Qdrant по интенту.

    Использует topic и year в запросе для лучшего матчинга.
    """
    from qdrant_client import QdrantClient

    # Формируем поисковый запрос: тема + год.
    parts: List[str] = []
    if intent.topic:
        parts.append(f"анекдот про {intent.topic}")
    else:
        parts.append("анекдот")
    if intent.year:
        parts.append(f"{intent.year} года")
    query = " ".join(parts)

    console.print(f"RAG-поиск в Qdrant: [italic]«{query}»[/]...")
    query_vec = embedder.encode(
        [query], convert_to_numpy=True, show_progress_bar=False
    )[0].tolist()

    client = QdrantClient(url=qdrant_url, timeout=QDRANT_TIMEOUT_SEC)
    # Опционально — фильтр по topic/year, если они есть.
    from qdrant_client.http import models as qm
    must: List[qm.FieldCondition] = []
    if intent.topic:
        must.append(qm.FieldCondition(
            key="topic",
            match=qm.MatchValue(value=intent.topic),
        ))
    if intent.year:
        must.append(qm.FieldCondition(
            key="year",
            match=qm.MatchValue(value=intent.year),
        ))
    flt = qm.Filter(must=must) if must else None

    # Сначала ищем с фильтром; если пусто — без фильтра.
    search_result = client.search(
        collection_name=collection,
        query_vector=query_vec,
        limit=top_k,
        with_payload=True,
        query_filter=flt,
    )
    if not search_result and flt is not None:
        console.print(
            "[yellow]С фильтром ничего не найдено, повтор без фильтра...[/]"
        )
        search_result = client.search(
            collection_name=collection,
            query_vector=query_vec,
            limit=top_k,
            with_payload=True,
        )

    hits: List[Dict[str, Any]] = []
    for point in search_result:
        payload = point.payload or {}
        hits.append({
            "id": str(payload.get("id", point.id)),
            "text": str(payload.get("text", "")),
            "topic": str(payload.get("topic", "")),
            "year": payload.get("year"),
            "score": float(point.score),
        })
    return hits


def rerank_jokes(
    query: str,
    hits: List[Dict[str, Any]],
    reranker_model: str,
) -> List[Dict[str, Any]]:
    """Переранжирует анекдоты через cross-encoder."""
    if not hits:
        return hits
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        console.print(
            "[yellow]sentence-transformers не установлен, пропускаем rerank.[/]"
        )
        return hits

    try:
        reranker = CrossEncoder(reranker_model)
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[yellow]Не удалось загрузить ранкер '{reranker_model}':[/] {exc}\n"
            f"   Пропускаем rerank."
        )
        return hits

    pairs = [[query, h["text"]] for h in hits]
    scores = reranker.predict(pairs).tolist()
    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)
    return sorted(hits, key=lambda h: h.get("rerank_score", 0.0), reverse=True)


# --- Шаг 3-alt: Лёгкий RAG через TF-IDF (fallback) -------------------------

def retrieve_jokes_light(
    intent: Intent,
    corpus_path: Path,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Лёгкий TF-IDF поиск анекдотов через retrieve_light.py.

    Срабатывает, когда sentence-transformers или Qdrant недоступны.
    Использует sklearn TfidfVectorizer + cosine_similarity + nltk Snowball
    (русский стемминг).
    """
    # Импортируем retrieve_light как модуль (он рядом).
    rag_dir = Path(__file__).resolve().parent
    if str(rag_dir) not in sys.path:
        sys.path.insert(0, str(rag_dir))
    try:
        import retrieve_light as rl  # type: ignore
    except ImportError as exc:
        console.print(
            f"[bold red]Ошибка:[/] не удалось импортировать retrieve_light.py: {exc}"
        )
        return []

    if not corpus_path.exists():
        console.print(f"[bold red]Ошибка:[/] корпус не найден: {corpus_path}")
        return []

    # Загружаем корпус и строим индекс.
    records = rl.load_corpus(corpus_path)
    vectorizer, tfidf_matrix = rl.build_tfidf_index(records)

    # Формируем поисковый запрос из интента.
    parts: List[str] = []
    if intent.topic:
        parts.append(f"анекдот про {intent.topic}")
    else:
        parts.append("анекдот")
    if intent.year:
        parts.append(f"{intent.year} года")
    query = " ".join(parts)
    console.print(f"RAG-light (TF-IDF) поиск: [italic]«{query}»[/]...")

    hits = rl.search_tfidf(
        query=query,
        records=records,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        top_k=top_k,
        topic_filter=intent.topic,
        year_filter=intent.year,
    )
    return [h.to_dict() for h in hits]


# --- Шаг 4: LLM-адаптация текста ------------------------------------------

ADAPT_SYSTEM_PROMPT: str = (
    "Ты адаптируешь советский анекдот 1980-х годов под устную подачу. "
    "Сделай начало естественным для устной речи: можно начать с "
    "'А вы слышали...' или 'Вот был такой случай...'. Не меняй сути шутки, "
    "не добавляй новых персонажей, не удлиняй более чем на 20%. "
    "Верни только итоговый текст анекдота, без пояснений."
)


def adapt_joke_text(
    joke_text: str,
    vllm_url: str,
    llm_model: str,
) -> Optional[str]:
    """Просит LLM адаптировать текст анекдота. Возвращает None при ошибке."""
    reply = call_vllm(
        user_text=f"Адаптируй под устную подачу:\n\n{joke_text}",
        vllm_url=vllm_url,
        model=llm_model,
        system_prompt=ADAPT_SYSTEM_PROMPT,
        temperature=0.5,
        max_tokens=512,
    )
    if reply is None:
        return None
    return reply.strip()


# --- Шаг 5: TTS ------------------------------------------------------------

def synth_tts(
    text: str,
    checkpoint_dir: Path,
    out_path: Path,
    language: str,
) -> Path:
    """Синтез речи через infer-логику из voice/tts."""
    # Импортируем infer.py как модуль (он рядом, в ../tts).
    tts_dir = (Path(__file__).resolve().parent.parent / "tts")
    sys.path.insert(0, str(tts_dir))
    try:
        import infer as infer_mod  # type: ignore
    except ImportError as exc:
        console.print(f"[bold red]Ошибка:[/] не удалось импортировать tts/infer.py: {exc}")
        sys.exit(4)

    # Готовим meta + speaker_ref и вызываем синтез напрямую.
    meta = infer_mod.load_checkpoint_meta(checkpoint_dir)
    speaker_ref = infer_mod.pick_speaker_ref(checkpoint_dir, meta)
    infer_mod.synthesize_xtts(
        text=text,
        checkpoint_dir=checkpoint_dir,
        speaker_ref=speaker_ref,
        language=language,
        out_path=out_path,
    )
    return out_path


# --- Печать результата -----------------------------------------------------

def print_result(result: PipelineResult) -> None:
    """Красиво печатает результат пайплайна."""
    console.print(Panel.fit("[bold]Результат пайплайна[/]", border_style="blue"))

    t = Table(show_header=False, box=None, padding=(0, 1))
    t.add_column(style="cyan", no_wrap=True)
    t.add_column()
    t.add_row("Ввод", f"{'[голос]' if result.asr_used else '[текст]'} {result.input_text}")
    t.add_row("Intent", str(result.intent.to_dict()))
    t.add_row("Найдено (top-5)", str(len(result.retrieved)))
    if result.selected:
        t.add_row("Выбран ID", result.selected.get("id", "—"))
        t.add_row("Тема/год", f"{result.selected.get('topic', '—')} / {result.selected.get('year', '—')}")
    t.add_row("Адаптирован", "да" if result.adapted_text else "нет")
    t.add_row("Аудио", str(result.audio_path) if result.audio_path else "—")
    console.print(t)

    if result.selected:
        console.print("\n[bold]Текст анекдота:[/]")
        console.print(f"  [italic]«{result.selected.get('text', '')}»[/]")
        if result.adapted_text:
            console.print("\n[bold]Адаптированный текст:[/]")
            console.print(f"  [italic]«{result.adapted_text}»[/]")

    if result.timings:
        tt = Table(title="Тайминги", show_header=True)
        tt.add_column("Шаг", style="cyan")
        tt.add_column("Время, сек", justify="right", style="magenta")
        for k, v in result.timings.items():
            tt.add_row(k, f"{v:.3f}")
        console.print(tt)

    if result.notes:
        console.print("\n[bold yellow]Замечания:[/]")
        for n in result.notes:
            console.print(f"  • {n}")


# --- Основной пайплайн ----------------------------------------------------

def run_pipeline(
    text: Optional[str],
    audio: Optional[Path],
    qdrant_url: str,
    collection: str,
    vllm_url: str,
    llm_model: str,
    embedding_model: str,
    reranker_model: str,
    tts_checkpoint: Path,
    out_dir: Path,
    top_k: int,
    whisper_model_name: str,
    use_rerank: bool,
    use_adapt: bool,
    use_tts: bool,
    language: str,
    corpus_path: Path = Path(DEFAULT_CORPUS_PATH),
) -> PipelineResult:
    """Запускает все шаги пайплайна."""
    console.print("[bold blue]== Полный пайплайн: ASR → Intent → RAG → TTS ==[/]")

    # Проверяем тяжёлые зависимости мягко: если их нет — переключаемся
    # на лёгкий TF-IDF RAG (retrieve_light.py).
    heavy_qdrant = _try_import("qdrant_client")
    heavy_st = _try_import("sentence_transformers")
    use_light_rag = not (heavy_qdrant and heavy_st)
    if use_light_rag:
        console.print(
            "[yellow]Внимание:[/] qdrant_client/sentence-transformers недоступны — "
            "RAG работает в лёгком режиме (TF-IDF, retrieve_light.py)."
        )
    if use_tts:
        _check_pkg("TTS", "TTS")
        _check_pkg("torch", "torch")

    out_dir.mkdir(parents=True, exist_ok=True)

    timings: Dict[str, float] = {}
    notes: List[str] = []
    if use_light_rag:
        notes.append("RAG: использован лёгкий TF-IDF режим (retrieve_light.py).")

    # --- Шаг 1: ASR (если голосовой ввод) ---
    asr_used = False
    if audio is not None:
        _check_pkg("whisper", "openai-whisper")
        t0 = time.time()
        text = transcribe_audio(audio, whisper_model_name)
        timings["asr"] = time.time() - t0
        asr_used = True
        console.print(f"ASR результат: [italic]«{text}»[/]")
    elif text is None:
        console.print(
            "[bold red]Ошибка:[/]"
            "нужно указать либо --text, либо --audio."
        )
        sys.exit(5)

    if not text or not text.strip():
        console.print("[bold red]Ошибка:[/] пустой текст запроса.")
        sys.exit(6)

    # --- Шаг 2: Извлечение интента ---
    t0 = time.time()
    intent = extract_intent(text, vllm_url=vllm_url, llm_model=llm_model)
    timings["intent"] = time.time() - t0

    # --- Шаг 3: RAG retrieve ---
    if use_light_rag:
        # Лёгкий путь: TF-IDF без Qdrant/sentence-transformers.
        t0 = time.time()
        retrieved = retrieve_jokes_light(
            intent=intent,
            corpus_path=corpus_path,
            top_k=top_k,
        )
        timings["rag_retrieve"] = time.time() - t0
        console.print(f"RAG-light: найдено [bold]{len(retrieved)}[/] анекдотов.")
    else:
        # Тяжёлый путь: sentence-transformers + Qdrant.
        console.print(f"Загрузка эмбеддера: {embedding_model}...")
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(embedding_model)

        t0 = time.time()
        retrieved = retrieve_jokes(
            intent=intent,
            qdrant_url=qdrant_url,
            collection=collection,
            embedder=embedder,
            top_k=top_k,
        )
        timings["rag_retrieve"] = time.time() - t0
        console.print(f"RAG: найдено [bold]{len(retrieved)}[/] анекдотов.")

    if not retrieved:
        console.print(
            "[bold red]Ошибка:[/]"
            "по запросу ничего не найдено. Возможно, индекс не собран "
            "или коллекция пуста."
        )
        sys.exit(7)

    # --- Шаг 4: Rerank (только в тяжёлом режиме) ---
    if use_rerank and not use_light_rag:
        console.print(f"Реранк через cross-encoder: {reranker_model}...")
        t0 = time.time()
        retrieved = rerank_jokes(
            query=intent.raw_query,
            hits=retrieved,
            reranker_model=reranker_model,
        )
        timings["rerank"] = time.time() - t0
    elif use_rerank and use_light_rag:
        notes.append("Rerank пропущен: требует sentence-transformers (недоступен).")

    selected = retrieved[0] if retrieved else None

    # --- Шаг 5: LLM-адаптация ---
    adapted_text: Optional[str] = None
    if use_adapt and selected:
        console.print("LLM-адаптация текста...")
        t0 = time.time()
        adapted_text = adapt_joke_text(
            selected["text"], vllm_url=vllm_url, llm_model=llm_model
        )
        timings["llm_adapt"] = time.time() - t0
        if adapted_text is None:
            notes.append("LLM-адаптация недоступна (vLLM оффлайн), используется оригинал.")

    final_text = adapted_text or (selected["text"] if selected else "")

    # --- Шаг 6: TTS ---
    audio_path: Optional[str] = None
    if use_tts and final_text:
        out_wav = out_dir / "result.wav"
        console.print(f"TTS: синтез → {out_wav}...")
        t0 = time.time()
        synth_tts(
            text=final_text,
            checkpoint_dir=tts_checkpoint,
            out_path=out_wav,
            language=language,
        )
        timings["tts"] = time.time() - t0
        audio_path = str(out_wav)

    # --- Сохраняем метаданные ---
    result = PipelineResult(
        input_text=text,
        asr_used=asr_used,
        intent=intent,
        retrieved=[{k: v for k, v in h.items() if k != "rerank_score"} or h for h in retrieved[:top_k]],
        selected=selected,
        adapted_text=adapted_text,
        audio_path=audio_path,
        timings=timings,
        notes=notes,
    )

    meta_path = out_dir / "result.json"
    meta_path.write_text(
        json.dumps(
            {
                "input_text": result.input_text,
                "asr_used": result.asr_used,
                "intent": result.intent.to_dict(),
                "selected": result.selected,
                "adapted_text": result.adapted_text,
                "audio_path": result.audio_path,
                "timings": result.timings,
                "notes": result.notes,
                "top_retrieved": result.retrieved,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print_result(result)
    return result


# --- CLI -------------------------------------------------------------------

@click.command()
@click.option(
    "--text", "text",
    type=str,
    default=None,
    help="Текстовый ввод запроса.",
)
@click.option(
    "--audio", "audio_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Аудиофайл (wav/mp3) для ASR. Если указан — переопределяет --text.",
)
@click.option(
    "--qdrant-url",
    type=str,
    default=DEFAULT_QDRANT_URL,
    show_default=True,
    help="URL Qdrant.",
)
@click.option(
    "--collection",
    type=str,
    default=DEFAULT_COLLECTION,
    show_default=True,
    help="Имя коллекции в Qdrant.",
)
@click.option(
    "--vllm-url",
    type=str,
    default=DEFAULT_VLLM_URL,
    show_default=True,
    help="URL vLLM (OpenAI-совместимый endpoint).",
)
@click.option(
    "--llm-model",
    type=str,
    default=DEFAULT_LLM_MODEL,
    show_default=True,
    help="Имя модели в vLLM.",
)
@click.option(
    "--embedding-model",
    type=str,
    default=DEFAULT_EMBEDDING_MODEL,
    show_default=True,
    help="Модель sentence-transformers для векторизации.",
)
@click.option(
    "--reranker-model",
    type=str,
    default=DEFAULT_RERANKER_MODEL,
    show_default=True,
    help="Модель cross-encoder для rerank.",
)
@click.option(
    "--tts-checkpoint",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path(DEFAULT_TTS_CHECKPOINT),
    show_default=True,
    help="Папка checkpoint TTS (результат train.py).",
)
@click.option(
    "--out", "out_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path(DEFAULT_OUT_DIR),
    show_default=True,
    help="Куда сохранять результат (wav + json).",
)
@click.option(
    "--top-k",
    type=int,
    default=DEFAULT_TOP_K,
    show_default=True,
    help="Сколько анекдотов доставать из RAG.",
)
@click.option(
    "--whisper-model",
    type=str,
    default=DEFAULT_WHISPER_MODEL,
    show_default=True,
    help="Размер Whisper: tiny|base|small|medium|large.",
)
@click.option(
    "--no-rerank",
    is_flag=True,
    default=False,
    help="Отключить rerank cross-encoder'ом.",
)
@click.option(
    "--no-adapt",
    is_flag=True,
    default=False,
    help="Отключить LLM-адаптацию текста анекдота.",
)
@click.option(
    "--no-tts",
    is_flag=True,
    default=False,
    help="Отключить финальный TTS (только поиск + адаптация).",
)
@click.option(
    "--language",
    type=str,
    default="ru",
    show_default=True,
    help="Язык синтеза TTS.",
)
@click.option(
    "--corpus", "corpus_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path(DEFAULT_CORPUS_PATH),
    show_default=True,
    help="Путь к jokes_corpus.jsonl (для лёгкого TF-IDF RAG, если Qdrant недоступен).",
)
def main(
    text: Optional[str],
    audio_path: Optional[Path],
    qdrant_url: str,
    collection: str,
    vllm_url: str,
    llm_model: str,
    embedding_model: str,
    reranker_model: str,
    tts_checkpoint: Path,
    out_dir: Path,
    top_k: int,
    whisper_model: str,
    no_rerank: bool,
    no_adapt: bool,
    no_tts: bool,
    language: str,
    corpus_path: Path,
) -> None:
    """Полный пайплайн: ввод → ASR → Intent (LLM) → RAG → Rerank → Adapt → TTS.

    Если sentence-transformers или Qdrant недоступны — RAG автоматически
    переключается на лёгкий TF-IDF поиск через retrieve_light.py.
    """
    run_pipeline(
        text=text,
        audio=audio_path,
        qdrant_url=qdrant_url,
        collection=collection,
        vllm_url=vllm_url,
        llm_model=llm_model,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        tts_checkpoint=tts_checkpoint,
        out_dir=out_dir,
        top_k=top_k,
        whisper_model_name=whisper_model,
        use_rerank=not no_rerank,
        use_adapt=not no_adapt,
        use_tts=not no_tts,
        language=language,
        corpus_path=corpus_path,
    )


if __name__ == "__main__":
    main()
