"""Построение индекса Qdrant по корпусу анекдотов 1986 года.

Скрипт читает jokes_corpus.jsonl, векторизует каждый анекдот моделью
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` и создаёт
коллекцию `jokes_1986` в Qdrant.

Запуск:
    python build_index.py \
        --corpus ../data/jokes_corpus.jsonl \
        --qdrant-url http://localhost:6333 \
        --collection jokes_1986

Если Qdrant не запущен — падает с понятной ошибкой на русском.
"""

from __future__ import annotations

# --- Импорты ---------------------------------------------------------------
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# --- Константы -------------------------------------------------------------

DEFAULT_CORPUS: str = "../data/jokes_corpus.jsonl"
DEFAULT_QDRANT_URL: str = "http://localhost:6333"
DEFAULT_COLLECTION: str = "jokes_1986"
DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM: int = 384  # размерность для paraphrase-multilingual-MiniLM-L12-v2
QDRANT_TIMEOUT_SEC: float = 5.0

console = Console()


# --- Утилиты ---------------------------------------------------------------

def _check_pkg(import_name: str, pip_name: str) -> None:
    """Проверяет наличие Python-пакета, иначе падает с понятной ошибкой."""
    try:
        __import__(import_name)
    except ImportError:
        console.print(f"[bold red]Ошибка:[/] Python-пакет '{pip_name}' не установлен.")
        console.print(f"   Установите: pip install {pip_name}")
        sys.exit(1)


def load_corpus(corpus_path: Path) -> List[Dict[str, Any]]:
    """Читает jokes_corpus.jsonl, возвращает список записей.

    Каждая запись: {"id": "j_0001", "year": 1986, "topic": "...", "text": "..."}
    """
    if not corpus_path.exists():
        console.print(
            f"[bold red]Ошибка:[/]"
            f"файл корпуса не найден: {corpus_path}"
        )
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
                    f"[yellow]Пропуск строки {line_no}:[/]"
                    f"невалидный JSON: {exc}"
                )
                continue
            # Проверяем обязательные поля.
            if not all(k in rec for k in ("id", "text")):
                console.print(
                    f"[yellow]Пропуск строки {line_no}:[/]"
                    f"нет обязательных полей id/text."
                )
                continue
            records.append(rec)

    if not records:
        console.print(
            f"[bold red]Ошибка:[/]"
            f"в {corpus_path} нет ни одной валидной записи."
        )
        sys.exit(3)

    return records


def check_qdrant(url: str) -> None:
    """Проверяет доступность Qdrant, иначе падает с понятной ошибкой."""
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import UnexpectedResponse

    try:
        client = QdrantClient(url=url, timeout=QDRANT_TIMEOUT_SEC)
        # Простой ping: получить список коллекций.
        _ = client.get_collections()
    except UnexpectedResponse as exc:
        console.print(
            f"[bold red]Qdrant вернул ошибку:[/] {exc}\n"
            f"   URL: {url}"
        )
        sys.exit(4)
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[bold red]Не удалось подключиться к Qdrant по адресу {url}.[/]\n"
            f"   Причина: {exc}\n"
            f"   Запустите Qdrant, например через Docker:\n"
            f"     docker run -d -p 6333:6333 qdrant/qdrant"
        )
        sys.exit(5)


def load_embedder(model_name: str):
    """Загружает sentence-transformers модель."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        console.print("[bold red]Ошибка:[/] sentence-transformers не установлен.")
        console.print("   Установите: pip install sentence-transformers")
        sys.exit(6)
    try:
        console.print(f"Загрузка модели эмбеддингов: {model_name}...")
        return SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[bold red]Не удалось загрузить модель '{model_name}'.[/]\n"
            f"   Причина: {exc}\n"
            f"   Возможные причины: нет интернета, мало места на диске, "
            f"неправильное имя модели."
        )
        sys.exit(7)


def build_texts_for_embedding(records: List[Dict[str, Any]]) -> List[str]:
    """Готовит тексты для векторизации.

    Добавляем тему и год в начало, чтобы эмбеддинг лучше «чувствовал»
    категорию анекдота.
    """
    texts: List[str] = []
    for r in records:
        topic = r.get("topic", "")
        year = r.get("year", "")
        body = r.get("text", "")
        header = f"Тема: {topic}. Год: {year}." if topic or year else ""
        text = f"{header} {body}".strip()
        texts.append(text)
    return texts


# --- Основная логика ------------------------------------------------------

def build_index(
    corpus_path: Path,
    qdrant_url: str,
    collection: str,
    model_name: str,
    recreate: bool,
) -> None:
    """Создаёт и наполняет коллекцию в Qdrant."""
    console.print("[bold blue]== Построение индекса Qdrant по анекдотам ==[/]")

    # Проверки зависимостей.
    _check_pkg("qdrant_client", "qdrant-client")
    _check_pkg("sentence_transformers", "sentence-transformers")

    # 1) Загружаем корпус.
    console.print(f"Чтение корпуса: {corpus_path}...")
    records = load_corpus(corpus_path)
    console.print(f"Записей в корпусе: [bold]{len(records)}[/]")

    # 2) Проверяем Qdrant.
    console.print(f"Проверка Qdrant: {qdrant_url}...")
    check_qdrant(qdrant_url)
    console.print("[green]Qdrant доступен.[/]")

    # 3) Загружаем эмбеддер.
    embedder = load_embedder(model_name)

    # 4) Векторизуем.
    texts = build_texts_for_embedding(records)
    console.print(f"Векторизация {len(texts)} текстов...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embed...", total=len(texts))
        # sentence-transformers умеет батчить; сделаем чанками по 32.
        embeddings: List[List[float]] = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vecs = embedder.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).tolist()
            embeddings.extend(vecs)
            progress.advance(task, advance=len(batch))

    # 5) Создаём / пересоздаём коллекцию.
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm

    client = QdrantClient(url=qdrant_url, timeout=QDRANT_TIMEOUT_SEC)

    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        if recreate:
            console.print(
                f"[yellow]Удаление существующей коллекции '{collection}'...[/]"
            )
            client.delete_collection(collection_name=collection)
            time.sleep(0.5)
        else:
            console.print(
                f"[bold red]Ошибка:[/]"
                f"коллекция '{collection}' уже существует. "
                f"Используйте --recreate для пересоздания."
            )
            sys.exit(8)

    console.print(f"Создание коллекции '{collection}' (dim={EMBEDDING_DIM})...")
    client.create_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(
            size=EMBEDDING_DIM,
            distance=qm.Distance.COSINE,
        ),
    )

    # 6) Заливаем точки.
    points: List[qm.PointStruct] = []
    for rec, vec in zip(records, embeddings):
        # Используем id из корпуса, если оно числовое; иначе хешируем в int.
        point_id = _to_qdrant_id(rec["id"])
        payload = {
            "id": rec["id"],
            "text": rec.get("text", ""),
            "topic": rec.get("topic", ""),
            "year": rec.get("year"),
        }
        # Сохраняем любые дополнительные поля.
        for k, v in rec.items():
            if k not in payload:
                payload[k] = v
        points.append(qm.PointStruct(id=point_id, vector=vec, payload=payload))

    # Заливаем чанками по 256.
    console.print(f"Заливаем {len(points)} точек в Qdrant...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Upload...", total=len(points))
        chunk = 256
        for i in range(0, len(points), chunk):
            client.upsert(
                collection_name=collection,
                points=points[i:i + chunk],
                wait=True,
            )
            progress.advance(task, advance=len(points[i:i + chunk]))

    # 7) Финальный отчёт.
    info = client.get_collection(collection_name=collection)
    table = Table(title=f"Коллекция '{collection}'", show_header=True)
    table.add_column("Параметр", style="cyan")
    table.add_column("Значение", style="magenta")
    table.add_row("Имя коллекции", collection)
    table.add_row("Размерность", str(EMBEDDING_DIM))
    table.add_row("Точек в коллекции", str(info.points_count))
    table.add_row("Дистанция", "Cosine")
    console.print(table)

    console.print(
        f"\n[bold green]Готово.[/] Индекс создан. "
        f"Теперь можно искать: python retrieve.py --query '...'"
    )


def _to_qdrant_id(rec_id: str) -> int:
    """Преобразует строковый id в uint для Qdrant (через hash)."""
    import hashlib
    h = hashlib.md5(rec_id.encode("utf-8")).hexdigest()
    # Qdrant принимает uint64; берём первые 16 hex-символов и режем до 8 байт.
    return int(h[:16], 16) % (2 ** 63)


# --- CLI -------------------------------------------------------------------

@click.command()
@click.option(
    "--corpus", "corpus_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path(DEFAULT_CORPUS),
    show_default=True,
    help="Путь к jokes_corpus.jsonl.",
)
@click.option(
    "--qdrant-url",
    type=str,
    default=DEFAULT_QDRANT_URL,
    show_default=True,
    help="URL Qdrant (например http://localhost:6333).",
)
@click.option(
    "--collection",
    type=str,
    default=DEFAULT_COLLECTION,
    show_default=True,
    help="Имя коллекции в Qdrant.",
)
@click.option(
    "--model", "model_name",
    type=str,
    default=DEFAULT_EMBEDDING_MODEL,
    show_default=True,
    help="Модель sentence-transformers для векторизации.",
)
@click.option(
    "--recreate",
    is_flag=True,
    default=False,
    help="Удалить и пересоздать коллекцию, если она существует.",
)
def main(
    corpus_path: Path,
    qdrant_url: str,
    collection: str,
    model_name: str,
    recreate: bool,
) -> None:
    """Построение индекса Qdrant по корпусу анекдотов 1986 года."""
    build_index(
        corpus_path=corpus_path,
        qdrant_url=qdrant_url,
        collection=collection,
        model_name=model_name,
        recreate=recreate,
    )


if __name__ == "__main__":
    main()
