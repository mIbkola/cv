"""Лёгкий поиск анекдотов на TF-IDF (без sentence-transformers / torch).

Эта версия — fallback для случая, когда sentence-transformers и torch
не ставятся (мало места на диске / timeout при установке).

Использует:
    - sklearn.feature_extraction.text.TfidfVectorizer (русский, ngram (1,2))
    - sklearn.metrics.pairwise.cosine_similarity

API совместим с retrieve.py:
    CLI:  python retrieve_light.py --query "анекдот про штирлица" --top-k 5
                              --corpus ../data/jokes_corpus.jsonl
                              [--topic штирлиц] [--year 1986]
                              [--json]

Зависимости (легковесные):
    pip install scikit-learn numpy rich click

Запуск:
    cd voice/rag
    python retrieve_light.py --query "анекдот про штирлица" --top-k 3

Если указать --json — дополнительно печатает в stdout чистый JSON-блок
{"query": ..., "results": [...]}, который удобно парсить из других скриптов.
"""

from __future__ import annotations

# --- Импорты ---------------------------------------------------------------
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# --- Константы -------------------------------------------------------------

DEFAULT_CORPUS: str = str(
    Path(__file__).resolve().parent.parent / "data" / "jokes_corpus.jsonl"
)
DEFAULT_TOP_K: int = 5
NGRAM_RANGE: tuple[int, int] = (1, 2)
MAX_FEATURES: int = 10000
MIN_DF: int = 1  # в маленьком корпусе игнорируем редкость слишком агрессивно
SUBLINEAR_TF: bool = True

console = Console()


# --- Модельки данных -------------------------------------------------------

@dataclass
class JokeHit:
    """Один результат поиска анекдота."""
    id: str
    text: str
    topic: str
    year: Optional[int]
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --- Токенизатор + стемминг для русского -----------------------------------

# Удаляем пунктуацию, оставляем буквы и дефис внутри слов.
_TOKEN_RE = re.compile(r"[А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z\-']*")

# Ленивая инициализация стеммера (nltk Snowball Russian).
# Стемминг приводит «штирлица»/«штирлицу» → «штирлиц», «брежнева» → «брежнев»
# и т.п. — это критично для матчинга запросов и текстов.
_stemmer = None


def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        try:
            from nltk.stem.snowball import RussianStemmer
            _stemmer = RussianStemmer()
        except ImportError:
            # Если nltk не установлен — fallback на «без стемминга».
            _stemmer = False
    return _stemmer


def russian_tokenize(text: str) -> List[str]:
    """Токенизатор + стемминг + фильтр стоп-слов.

    Разбивает на слова, стеммит, нижний регистр, выкидывает стоп-слова.

    sklearn TfidfVectorizer по умолчанию плохо работает с кириллицей
    (его default token_pattern = r\"(?u)\\b\\w\\w+\\b\" — кириллицу ест,
    но не даёт гибкости). Поэтому используем свой tokenizer.

    Стемминг нужен, чтобы запрос «анекдот про штирлица» матился на
    «штирлиц» в текстах. Без стемминга cosine_similarity будет почти
    нулевой для большинства русских запросов.

    Стоп-слова фильтруем ВНУТРИ токенизатора (а не через параметр
    stop_words у TfidfVectorizer) — это позволяет избежать проблемы
    двойного стемминга: sklearn применяет tokenizer к каждому stop_word,
    что для уже-простемленных слов даёт мусор (например, «опя» → «оп»).
    """
    text = text.lower().replace("ё", "е")
    tokens = _TOKEN_RE.findall(text)
    stemmer = _get_stemmer()
    if stemmer:
        out: List[str] = []
        for tok in tokens:
            stemmed = stemmer.stem(tok)
            if stemmed not in RUSSIAN_STOPWORDS:
                out.append(stemmed)
        return out
    # Без стеммера — фильтруем как есть.
    return [t for t in tokens if t not in RUSSIAN_STOPWORDS]


# Стоп-слова для русского: убираем частые бессмысленные + типовые слова-запросы.
# ВАЖНО: сюда НЕ входят содержательные слова персонажей (василь, иваныч, петька,
# однако, штирлиц, чукча) — они нужны для матчинга.
_RAW_STOPWORDS: tuple[str, ...] = (
    # общие местоимения/союзы/предлоги
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
    "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к",
    "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было",
    "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь",
    "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни",
    "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам",
    "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они",
    "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя", "их",
    "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти",
    "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем",
    "всех", "никогда", "можно", "при", "наконец", "два", "об", "другой",
    "хоть", "после", "над", "много", "того", "лишь",
    # типовые слова запроса (не содержательные)
    "анекдот", "про", "расскажи", "пошути", "шутка", "история", "байка",
    "год", "года", "годы", "тот", "которые", "которая", "какой-то",
)


def _build_stopwords() -> frozenset[str]:
    """Строит множество стоп-слов с учётом стемминга.

    Если nltk доступен — применяем стемминг к каждому стоп-слову, чтобы
    они матились с простемленными токенами документов.
    """
    stemmer = _get_stemmer()
    if stemmer:
        # Двойной стемминг для устойчивости: «опять» → «опя» → «оп».
        # Так стоп-слова будут сравнимы с любым возможным результатом стемминга.
        single = frozenset(stemmer.stem(w) for w in _RAW_STOPWORDS)
        double = frozenset(stemmer.stem(stemmer.stem(w)) for w in _RAW_STOPWORDS)
        return single | double
    return frozenset(_RAW_STOPWORDS)


# Инициализируем лениво — при первом вызове russian_tokenize.
# ВАЖНО: это должно происходить ДО первого вызова russian_tokenize,
# поэтому инициализируем сразу при импорте (после объявления _get_stemmer).
RUSSIAN_STOPWORDS: frozenset[str] = _build_stopwords()


# --- Загрузка корпуса ------------------------------------------------------

def load_corpus(corpus_path: Path) -> List[Dict[str, Any]]:
    """Читает jokes_corpus.jsonl. Каждая запись: {id, year, topic, text}."""
    if not corpus_path.exists():
        console.print(f"[bold red]Ошибка:[/] файл корпуса не найден: {corpus_path}")
        sys.exit(2)

    records: List[Dict[str, Any]] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                console.print(
                    f"[yellow]Пропуск строки {line_no}:[/] невалидный JSON: {exc}"
                )
                continue
            if not all(k in rec for k in ("id", "text")):
                console.print(
                    f"[yellow]Пропуск строки {line_no}:[/] нет полей id/text."
                )
                continue
            records.append(rec)

    if not records:
        console.print(f"[bold red]Ошибка:[/] в {corpus_path} нет валидных записей.")
        sys.exit(3)

    return records


# --- Построение TF-IDF индекса --------------------------------------------

def build_tfidf_index(records: List[Dict[str, Any]]):
    """Строит TfidfVectorizer и матрицу TF-IDF по корпусу.

    Возвращает (vectorizer, tfidf_matrix).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Готовим «обогащённый» текст: тема + год + сам текст анекдота.
    # Это даёт TF-IDF матчинг по теме, даже если в запросе не упомянули слова из text.
    docs: List[str] = []
    for r in records:
        topic = str(r.get("topic", "") or "")
        year = str(r.get("year", "") or "")
        body = str(r.get("text", "") or "")
        docs.append(f"тема: {topic} год: {year} {body}")

    vectorizer = TfidfVectorizer(
        tokenizer=russian_tokenize,
        # stop_words НЕ передаём — фильтруем внутри russian_tokenize,
        # иначе sklearn применит tokenizer к каждому stop_word и получит
        # двойной стемминг (см. комментарий в russian_tokenize).
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        sublinear_tf=SUBLINEAR_TF,
        token_pattern=None,  # отключаем дефолт, используем tokenizer
    )
    matrix = vectorizer.fit_transform(docs)
    return vectorizer, matrix


# --- Поиск -----------------------------------------------------------------

def search_tfidf(
    query: str,
    records: List[Dict[str, Any]],
    vectorizer,
    tfidf_matrix,
    top_k: int,
    topic_filter: Optional[str] = None,
    year_filter: Optional[int] = None,
) -> List[JokeHit]:
    """Ищет top-k анекдотов по косинусной близости TF-IDF векторов.

    Опционально фильтрует по topic/year ДО косинусной сортировки
    (если фильтр задан и в отфильтрованном подмножестве есть хоть один).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Векторизуем запрос тем же vectorizer'ом.
    query_vec = vectorizer.transform([query])

    # Считаем косинус к каждому документу.
    sims = cosine_similarity(query_vec, tfidf_matrix)[0]  # shape: (N,)

    # Применяем фильтры: формируем маску.
    mask: List[bool] = [True] * len(records)
    if topic_filter:
        tf_lower = topic_filter.lower().strip()
        mask = [
            m and str(r.get("topic", "")).lower() == tf_lower
            for r, m in zip(records, mask)
        ]
        # Если точное совпадение темы ничего не дало — пробуем частичное.
        if not any(mask):
            mask = [
                m and tf_lower in str(r.get("topic", "")).lower()
                for r, m in zip(records, mask)
            ]
    if year_filter is not None:
        mask = [
            m and (r.get("year") == year_filter)
            for r, m in zip(records, mask)
        ]

    # Если после фильтров ничего не осталось — сбрасываем фильтры
    # (лучше вернуть что-то, чем пустоту).
    if not any(mask):
        console.print(
            "[yellow]С фильтром ничего не найдено — повтор без фильтра...[/]"
        )
        mask = [True] * len(records)

    # Собираем hits с маской.
    scored: List[tuple[float, int]] = []
    for i, (sim, ok) in enumerate(zip(sims, mask)):
        if ok:
            scored.append((float(sim), i))
    scored.sort(key=lambda x: x[0], reverse=True)

    hits: List[JokeHit] = []
    for sim, i in scored[:top_k]:
        r = records[i]
        hits.append(JokeHit(
            id=str(r.get("id", "")),
            text=str(r.get("text", "")),
            topic=str(r.get("topic", "")),
            year=r.get("year"),
            score=sim,
        ))
    return hits


# --- Печать результата -----------------------------------------------------

def print_results(query: str, hits: List[JokeHit]) -> None:
    """Красиво печатает результаты через rich."""
    console.print(
        Panel.fit(
            f"[bold]Запрос:[/] {query}\n"
            f"[bold]Найдено:[/] {len(hits)} анекдотов",
            border_style="cyan",
        )
    )

    for i, h in enumerate(hits, 1):
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column()
        table.add_row("Ранг", f"#{i}")
        table.add_row("ID", h.id)
        table.add_row("Тема", h.topic or "—")
        table.add_row("Год", str(h.year) if h.year else "—")
        table.add_row("Score", f"{h.score:.4f}")
        table.add_row("Текст", "")
        console.print(table)
        console.print(f"  [italic]«{h.text}»[/]")
        console.print()


def print_json_block(query: str, hits: List[JokeHit], top_k: int) -> None:
    """Печатает чистый JSON-блок в stdout для парсинга другими скриптами."""
    payload = {
        "query": query,
        "top_k": top_k,
        "engine": "tfidf",
        "results": [h.to_dict() for h in hits],
    }
    print(json.dumps(payload, ensure_ascii=False))


# --- CLI -------------------------------------------------------------------

@click.command()
@click.option(
    "--query", "query",
    type=str,
    required=True,
    help="Текстовый запрос (например: 'анекдот про штирлица').",
)
@click.option(
    "--corpus", "corpus_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path(DEFAULT_CORPUS),
    show_default=True,
    help="Путь к jokes_corpus.jsonl.",
)
@click.option(
    "--top-k",
    type=int,
    default=DEFAULT_TOP_K,
    show_default=True,
    help="Сколько результатов вернуть.",
)
@click.option(
    "--topic", "topic_filter",
    type=str,
    default=None,
    help="Фильтр по теме (например: штирлиц, чукча, брежнев).",
)
@click.option(
    "--year", "year_filter",
    type=int,
    default=None,
    help="Фильтр по году (например: 1986).",
)
@click.option(
    "--json", "as_json",
    is_flag=True,
    default=False,
    help="Печатать чистый JSON-блок в stdout (для парсинга другими скриптами).",
)
def main(
    query: str,
    corpus_path: Path,
    top_k: int,
    topic_filter: Optional[str],
    year_filter: Optional[int],
    as_json: bool,
) -> None:
    """Лёгкий TF-IDF поиск анекдотов (без sentence-transformers и torch)."""
    # Проверяем зависимости.
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
        from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401
    except ImportError:
        console.print(
            "[bold red]Ошибка:[/] scikit-learn не установлен.\n"
            "   Установите: pip install scikit-learn numpy"
        )
        sys.exit(1)

    # Если --json, глушим лишний rich-вывод (он поедет в stderr? нет, в stdout).
    # Поэтому при --json мы печатаем только JSON-блок, без rich-таблиц.
    if not as_json:
        console.print("[bold blue]== Лёгкий TF-IDF поиск анекдотов ==[/]")
        console.print(f"Корпус: {corpus_path}")
        console.print(
            f"Параметры: ngram={NGRAM_RANGE}, max_features={MAX_FEATURES}, "
            f"sublinear_tf={SUBLINEAR_TF}"
        )

    # 1) Загружаем корпус.
    if not as_json:
        console.print(f"Чтение корпуса: {corpus_path}...")
    records = load_corpus(corpus_path)
    if not as_json:
        console.print(f"Записей в корпусе: [bold]{len(records)}[/]")

    # 2) Строим индекс.
    t0 = time.time()
    vectorizer, tfidf_matrix = build_tfidf_index(records)
    dt_index = time.time() - t0
    if not as_json:
        console.print(
            f"Индекс построен за [bold]{dt_index:.3f} сек[/], "
            f"матрица: {tfidf_matrix.shape[0]} × {tfidf_matrix.shape[1]}"
        )

    # 3) Поиск.
    if not as_json:
        console.print(
            f"Поиск top-{top_k} по запросу: [italic]«{query}»[/]..."
            + (f" (topic={topic_filter})" if topic_filter else "")
            + (f" (year={year_filter})" if year_filter else "")
        )
    t0 = time.time()
    hits = search_tfidf(
        query=query,
        records=records,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        top_k=top_k,
        topic_filter=topic_filter,
        year_filter=year_filter,
    )
    dt_search = time.time() - t0
    if not as_json:
        console.print(f"Поиск занял: [bold]{dt_search:.4f} сек[/]")

    # 4) Печать.
    if as_json:
        print_json_block(query, hits, top_k)
    else:
        print_results(query, hits)
        if not hits:
            console.print("[yellow]Ничего не найдено.[/]")


if __name__ == "__main__":
    main()
