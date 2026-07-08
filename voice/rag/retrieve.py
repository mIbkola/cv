"""Поиск анекдотов в Qdrant.

Скрипт принимает текстовый запрос, векторизует его, ищет top-k релевантных
анекдотов в Qdrant. Опционально — rerank через cross-encoder.

Запуск:
    python retrieve.py \
        --query "анекдот про штирлица" \
        --top-k 5 \
        --rerank

Если Qdrant не запущен — падает с понятной ошибкой.
"""

from __future__ import annotations

# --- Импорты ---------------------------------------------------------------
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

DEFAULT_QDRANT_URL: str = "http://localhost:6333"
DEFAULT_COLLECTION: str = "jokes_1986"
DEFAULT_TOP_K: int = 5
DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QDRANT_TIMEOUT_SEC: float = 5.0

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
    rerank_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --- Утилиты ---------------------------------------------------------------

def _check_pkg(import_name: str, pip_name: str) -> None:
    """Проверяет наличие Python-пакета, иначе падает с понятной ошибкой."""
    try:
        __import__(import_name)
    except ImportError:
        console.print(f"[bold red]Ошибка:[/] Python-пакет '{pip_name}' не установлен.")
        console.print(f"   Установите: pip install {pip_name}")
        sys.exit(1)


def check_qdrant(url: str) -> None:
    """Проверяет доступность Qdrant."""
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import UnexpectedResponse

    try:
        client = QdrantClient(url=url, timeout=QDRANT_TIMEOUT_SEC)
        _ = client.get_collections()
    except UnexpectedResponse as exc:
        console.print(f"[bold red]Qdrant вернул ошибку:[/] {exc}")
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[bold red]Не удалось подключиться к Qdrant по адресу {url}.[/]\n"
            f"   Причина: {exc}\n"
            f"   Запустите Qdrant:\n"
            f"     docker run -d -p 6333:6333 qdrant/qdrant\n"
            f"   и соберите индекс:\n"
            f"     python build_index.py"
        )
        sys.exit(3)


def load_embedder(model_name: str):
    """Загружает sentence-transformers модель."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        console.print("[bold red]Ошибка:[/] sentence-transformers не установлен.")
        sys.exit(4)
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[bold red]Не удалось загрузить модель '{model_name}'.[/]\n"
            f"   Причина: {exc}"
        )
        sys.exit(5)


def load_reranker(model_name: str):
    """Загружает cross-encoder ранкер."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        console.print("[bold red]Ошибка:[/] sentence-transformers не установлен.")
        sys.exit(6)
    try:
        return CrossEncoder(model_name)
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[bold red]Не удалось загрузить ранкер '{model_name}'.[/]\n"
            f"   Причина: {exc}\n"
            f"   Установите pip install sentence-transformers"
        )
        sys.exit(7)


# --- Поиск -----------------------------------------------------------------

def search_qdrant(
    query: str,
    qdrant_url: str,
    collection: str,
    embedder,
    top_k: int,
) -> List[JokeHit]:
    """Векторизует запрос и ищет top-k в Qdrant."""
    from qdrant_client import QdrantClient

    client = QdrantClient(url=qdrant_url, timeout=QDRANT_TIMEOUT_SEC)

    # Векторизуем запрос.
    query_vec = embedder.encode(
        [query], convert_to_numpy=True, show_progress_bar=False
    )[0].tolist()

    # Проверяем, существует ли коллекция.
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        console.print(
            f"[bold red]Ошибка:[/]"
            f"коллекция '{collection}' не найдена в Qdrant.\n"
            f"   Сначала соберите индекс:\n"
            f"     python build_index.py --collection {collection}"
        )
        sys.exit(8)

    # Поиск.
    from qdrant_client.http import models as qm
    search_result = client.search(
        collection_name=collection,
        query_vector=query_vec,
        limit=top_k,
        with_payload=True,
    )

    hits: List[JokeHit] = []
    for point in search_result:
        payload = point.payload or {}
        hits.append(JokeHit(
            id=str(payload.get("id", point.id)),
            text=str(payload.get("text", "")),
            topic=str(payload.get("topic", "")),
            year=payload.get("year"),
            score=float(point.score),
        ))
    return hits


def rerank_hits(
    query: str,
    hits: List[JokeHit],
    reranker,
) -> List[JokeHit]:
    """Переранжирует hits через cross-encoder."""
    if not hits:
        return hits

    pairs = [[query, h.text] for h in hits]
    scores = reranker.predict(pairs).tolist()
    for h, s in zip(hits, scores):
        h.rerank_score = float(s)

    # Сортируем по убыванию rerank_score.
    return sorted(hits, key=lambda h: h.rerank_score or 0.0, reverse=True)


def print_results(query: str, hits: List[JokeHit]) -> None:
    """Красиво печатает результаты поиска через rich."""
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
        if h.rerank_score is not None:
            table.add_row("Rerank", f"{h.rerank_score:.4f}")
        table.add_row("Текст", "")
        console.print(table)
        console.print(f"  [italic]«{h.text}»[/]")
        console.print()


# --- CLI -------------------------------------------------------------------

@click.command()
@click.option(
    "--query", "query",
    type=str,
    required=True,
    help="Текстовый запрос (например: 'анекдот про штирлица').",
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
    "--top-k",
    type=int,
    default=DEFAULT_TOP_K,
    show_default=True,
    help="Сколько результатов вернуть.",
)
@click.option(
    "--rerank",
    is_flag=True,
    default=False,
    help="Включить переранжирование cross-encoder'ом.",
)
@click.option(
    "--reranker-model",
    type=str,
    default=DEFAULT_RERANKER_MODEL,
    show_default=True,
    help="Модель cross-encoder для rerank.",
)
@click.option(
    "--model", "model_name",
    type=str,
    default=DEFAULT_EMBEDDING_MODEL,
    show_default=True,
    help="Модель sentence-transformers для векторизации запроса.",
)
@click.option(
    "--out", "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Если указано — сохранить результаты в JSON.",
)
def main(
    query: str,
    qdrant_url: str,
    collection: str,
    top_k: int,
    rerank: bool,
    reranker_model: str,
    model_name: str,
    out_path: Optional[Path],
) -> None:
    """Поиск анекдотов в Qdrant по текстовому запросу."""
    import json

    console.print("[bold blue]== Поиск анекдотов в Qdrant ==[/]")

    # Проверки.
    _check_pkg("qdrant_client", "qdrant-client")
    _check_pkg("sentence_transformers", "sentence-transformers")

    # 1) Проверяем Qdrant.
    console.print(f"Проверка Qdrant: {qdrant_url}...")
    check_qdrant(qdrant_url)
    console.print("[green]Qdrant доступен.[/]")

    # 2) Загружаем эмбеддер.
    embedder = load_embedder(model_name)

    # 3) Поиск.
    console.print(f"Поиск top-{top_k} по запросу: [italic]«{query}»[/]...")
    t0 = time.time()
    hits = search_qdrant(
        query=query,
        qdrant_url=qdrant_url,
        collection=collection,
        embedder=embedder,
        top_k=top_k,
    )
    dt_search = time.time() - t0
    console.print(f"Поиск занял: [bold]{dt_search:.3f} сек[/]")

    # 4) Реранк (опционально).
    if rerank and hits:
        console.print(f"Реранк через cross-encoder: {reranker_model}...")
        reranker = load_reranker(reranker_model)
        t0 = time.time()
        hits = rerank_hits(query, hits, reranker)
        dt_rerank = time.time() - t0
        console.print(f"Реранк занял: [bold]{dt_rerank:.3f} сек[/]")

    # 5) Печать.
    print_results(query, hits)

    # 6) Сохранение в JSON, если попросили.
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "query": query,
                    "top_k": top_k,
                    "rerank": rerank,
                    "results": [h.to_dict() for h in hits],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        console.print(f"Результаты сохранены в: {out_path}")

    if not hits:
        console.print("[yellow]Ничего не найдено.[/]")


if __name__ == "__main__":
    main()
