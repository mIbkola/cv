#!/usr/bin/env python3
"""finetune.py — fine-tune YOLOv8 на датасете стаканов.

Принимает путь к ``dataset.yaml`` (формат ultralytics), запускает ``model.train()``.
Все параметры обучения передаются через CLI. Если ``ultralytics`` не установлен —
печатается понятное сообщение об ошибке.

Пример ``dataset.yaml``::

    path: /data/coffee_cups
    train: images/train
    val: images/val
    names:
      0: paper_cup

Пример запуска::

    python finetune.py --data dataset.yaml --model yolov8s.pt \\
        --epochs 100 --imgsz 640 --batch 16 --name coffee_cup_v1
"""

from __future__ import annotations

# --- Импорты ---
import sys
from pathlib import Path

import click
from rich.console import Console

# --- Константы ---
DEFAULT_MODEL = "yolov8s.pt"
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_DEVICE = "0"  # GPU по умолчанию; "cpu" для CPU

console = Console()


def _import_ultralytics():
    """Ленивый импорт ultralytics с понятным сообщением об ошибке."""
    try:
        from ultralytics import YOLO  # type: ignore
        return YOLO
    except ImportError as exc:  # pragma: no cover
        console.print(
            "[red]Ошибка: библиотека ultralytics не установлена.[/red]\n"
            "Установите: [cyan]pip install ultralytics>=8.0.0[/cyan]"
        )
        raise SystemExit(2) from exc


def _validate_dataset_yaml(path: Path) -> None:
    """Проверяет, что dataset.yaml существует и читается."""
    if not path.exists():
        console.print(f"[red]Ошибка: файл датасета не найден: {path}[/red]")
        raise SystemExit(2)
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        console.print(
            "[red]Ошибка: библиотека PyYAML не установлена.[/red]\n"
            "Установите: [cyan]pip install pyyaml[/cyan]"
        )
        raise SystemExit(2) from exc
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        console.print(f"[red]Ошибка парсинга YAML {path}: {exc}[/red]")
        raise SystemExit(2)
    for key in ("path", "train", "val", "names"):
        if key not in data:
            console.print(
                f"[red]Ошибка: в {path} отсутствует обязательный ключ '{key}'.[/red]\n"
                f"Ожидаемые ключи: path, train, val, names."
            )
            raise SystemExit(2)
    console.print(f"[green]dataset.yaml валиден: {len(data['names'])} класс(а/ов).[/green]")


def run_finetune(
    data_yaml: Path,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: str | None,
    name: str,
    patience: int,
    weights_only: bool,
) -> None:
    """Запускает ``model.train()`` с заданными параметрами."""
    YOLO = _import_ultralytics()
    _validate_dataset_yaml(data_yaml)

    console.print(f"[cyan]Загрузка базовой модели {model_name} ...[/cyan]")
    try:
        model = YOLO(model_name)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Не удалось загрузить модель {model_name}: {exc}[/red]")
        raise SystemExit(3)

    console.print(f"[cyan]Старт fine-tune на {data_yaml} ...[/cyan]")
    console.print(
        f"epochs={epochs}, imgsz={imgsz}, batch={batch}, device={device}, "
        f"patience={patience}, name={name}"
    )

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        project=project or "runs/detect",
        name=name,
        # Аугментации под бумажные стаканы: освещение, ракурс, лёгкий blur
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,  # стакан нельзя переворачивать
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )
    if weights_only:
        train_kwargs["weights_only"] = True

    try:
        results = model.train(**train_kwargs)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Ошибка во время обучения: {exc}[/red]")
        console.print_exception()
        raise SystemExit(4)

    # Лучшие веса обычно сохраняются в <project>/<name>/weights/best.pt
    save_dir = Path(project or "runs/detect") / name
    best = save_dir / "weights" / "best.pt"
    console.print(
        f"\n[green]Обучение завершено.[/green]\n"
        f"Каталог: {save_dir}\n"
        f"Лучшие веса: {best} "
        f"({'найден' if best.exists() else 'не найден — проверьте лог обучения'})"
    )
    try:
        # ultralytics возвращает объект с метриками
        if hasattr(results, "results_dict"):
            console.print(f"Финальные метрики: {results.results_dict}")
    except Exception:  # noqa: BLE001
        pass


@click.command()
@click.option("--data", "-d", "data_yaml", required=True, type=click.Path(exists=True),
              help="Путь к dataset.yaml (формат ultralytics).")
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Базовая модель YOLOv8.")
@click.option("--epochs", "-e", default=DEFAULT_EPOCHS, type=int, help="Число эпох.")
@click.option("--imgsz", default=DEFAULT_IMGSZ, type=int, help="Размер изображения.")
@click.option("--batch", "-b", default=DEFAULT_BATCH, type=int, help="Размер батча.")
@click.option("--device", default=DEFAULT_DEVICE, help="Устройство ('0', '0,1', 'cpu').")
@click.option("--project", default=None, help="Каталог проекта (по умолчанию runs/detect).")
@click.option("--name", default="coffee_cup", help="Имя запуска.")
@click.option("--patience", default=20, type=int, help="Ранняя остановка (эпохи без улучшения).")
@click.option("--weights-only", is_flag=True, default=False,
              help="Передать weights_only=True (только веса, не нужны оригинальные классы).")
def main(data_yaml: str, model: str, epochs: int, imgsz: int, batch: int,
         device: str, project: str | None, name: str, patience: int,
         weights_only: bool) -> None:
    """Fine-tune YOLOv8 на датасете бумажных стаканов."""
    console.print("[bold cyan]COFFEE detection — fine-tune YOLOv8[/bold cyan]")
    try:
        run_finetune(
            data_yaml=Path(data_yaml),
            model_name=model,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            patience=patience,
            weights_only=weights_only,
        )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Непредвиденная ошибка: {exc}[/red]")
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
