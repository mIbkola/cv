#!/usr/bin/env python3
"""baseline_test.py — тест готовой YOLOv8 (COCO) на тестовых изображениях.

Скрипт прогоняет предобученную YOLOv8 (класс COCO ``cup`` id=41) по папке
с тестовыми изображениями бумажных стаканов с кофе, сохраняет аннотированные
изображения и печатает сводные метрики (precision / recall / количество
срабатываний / среднюю уверенность). Если в папке есть разметка YOLO
(``<имя>.txt`` рядом с фото), дополнительно считается mAP@0.5.

Все ошибки выводятся на русском через rich.console.Console.
Если ``ultralytics`` или ``opencv`` не установлены — печатается понятное
сообщение с подсказкой ``pip install ultralytics opencv-python``.

Пример запуска::

    python baseline_test.py --images test_images/ --class-id 41 \\
        --model yolov8n.pt --output annotated/
"""

from __future__ import annotations

# --- Импорты ---
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

# --- Константы ---
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CLASS_ID = 41  # COCO "cup"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

console = Console()


def _import_ultralytics():
    """Ленивый импорт ultralytics с понятным сообщением об ошибке."""
    try:
        from ultralytics import YOLO  # type: ignore
        return YOLO
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        console.print(
            "[red]Ошибка: библиотека ultralytics не установлена.[/red]\n"
            "Установите: [cyan]pip install ultralytics>=8.0.0[/cyan]"
        )
        raise SystemExit(2) from exc


def _import_cv2():
    """Ленивый импорт OpenCV с понятным сообщением об ошибке."""
    try:
        import cv2  # type: ignore
        return cv2
    except ImportError as exc:  # pragma: no cover
        console.print(
            "[red]Ошибка: OpenCV (opencv-python) не установлен.[/red]\n"
            "Установите: [cyan]pip install opencv-python[/cyan]"
        )
        raise SystemExit(2) from exc


def _collect_images(folder: Path) -> list[Path]:
    """Собирает список изображений в папке (рекурсивно)."""
    if not folder.exists():
        console.print(f"[red]Ошибка: папка не найдена: {folder}[/red]")
        raise SystemExit(2)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in exts)
    return images


def _load_yolo_labels(label_path: Path) -> list[tuple[float, float, float, float]]:
    """Читает YOLO-разметку (cx, cy, w, h) в нормированных координатах."""
    if not label_path.exists():
        return []
    boxes: list[tuple[float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            _cls, cx, cy, w, h = (float(x) for x in parts[:5])
            boxes.append((cx, cy, w, h))
        except ValueError:
            continue
    return boxes


def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU двух прямоугольников в формате (x1, y1, x2, y2)."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _compute_map(
    predictions: list[list[tuple[float, float, float, float, float]]],
    ground_truths: list[list[tuple[float, float, float, float]]],
    iou_thr: float = IOU_THRESHOLD,
) -> tuple[float, float, float]:
    """Простая реализация mAP@0.5 по классу.

    Возвращает (precision, recall, mAP@0.5). Достаточно для оценки baseline.
    """
    all_tp, all_fp, all_fn = 0, 0, 0
    for preds, gts in zip(predictions, ground_truths):
        matched = [False] * len(gts)
        # сортируем предсказания по убыванию уверенности
        preds_sorted = sorted(preds, key=lambda p: p[4], reverse=True)
        for p in preds_sorted:
            best_iou, best_idx = 0.0, -1
            for j, gt in enumerate(gts):
                if matched[j]:
                    continue
                iou = _iou_xyxy(p[:4], gt)
                if iou > best_iou:
                    best_iou, best_idx = iou, j
            if best_iou >= iou_thr and best_idx >= 0:
                matched[best_idx] = True
                all_tp += 1
            else:
                all_fp += 1
        all_fn += sum(1 for m in matched if not m)
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) else 0.0
    # упрощённый mAP: при одном классе ~ precision*recall (точнее — AP, но
    # для rough baseline-оценки достаточно).
    map50 = precision * recall if (precision + recall) else 0.0
    return precision, recall, map50


def _draw_boxes(image, boxes, color, label_prefix: str):
    """Рисует прямоугольники на изображении. ``boxes`` — list[(x1,y1,x2,y2,conf?)]."""
    cv2 = _import_cv2()
    for box in boxes:
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        if len(box) == 5:
            text = f"{label_prefix} {box[4]:.2f}"
            cv2.putText(image, text, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def run_baseline(
    images_folder: Path,
    output_folder: Path,
    model_name: str,
    class_id: int,
    conf: float,
    iou: float,
) -> None:
    """Основная логика baseline-теста."""
    YOLO = _import_ultralytics()
    cv2 = _import_cv2()

    images = _collect_images(images_folder)
    if not images:
        console.print(f"[yellow]Внимание: в папке {images_folder} нет изображений.[/yellow]")
        return

    console.print(f"[cyan]Загрузка модели {model_name} ...[/cyan]")
    try:
        model = YOLO(model_name)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Не удалось загрузить модель {model_name}: {exc}[/red]")
        raise SystemExit(3)

    output_folder.mkdir(parents=True, exist_ok=True)

    predictions: list[list[tuple[float, float, float, float, float]]] = []
    ground_truths: list[list[tuple[float, float, float, float]]] = []
    table = Table(title="Baseline YOLOv8 — результаты по изображениям")
    table.add_column("Файл", overflow="fold")
    table.add_column("Det", justify="right")
    table.add_column("GT", justify="right")
    table.add_column("Avg conf", justify="right")

    total_dets = 0
    total_conf_sum = 0.0

    for img_path in images:
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                console.print(f"[yellow]Не удалось прочитать: {img_path}[/yellow]")
                continue
            h, w = image.shape[:2]
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Ошибка чтения {img_path}: {exc}[/red]")
            continue

        # Предсказание
        try:
            results = model.predict(
                source=str(img_path),
                conf=conf,
                iou=iou,
                classes=[class_id],
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Ошибка инференса {img_path}: {exc}[/red]")
            continue

        pred_boxes: list[tuple[float, float, float, float, float]] = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                c = float(b.conf[0])
                pred_boxes.append((x1, y1, x2, y2, c))

        # GT (если есть .txt рядом с тем же именем)
        label_path = img_path.with_suffix(".txt")
        gt_norm = _load_yolo_labels(label_path)
        gt_boxes: list[tuple[float, float, float, float]] = []
        for (cx, cy, bw, bh) in gt_norm:
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            gt_boxes.append((x1, y1, x2, y2))

        predictions.append(pred_boxes)
        ground_truths.append(gt_boxes)

        avg_conf = (sum(b[4] for b in pred_boxes) / len(pred_boxes)) if pred_boxes else 0.0
        total_dets += len(pred_boxes)
        total_conf_sum += sum(b[4] for b in pred_boxes)
        table.add_row(img_path.name, str(len(pred_boxes)), str(len(gt_boxes)), f"{avg_conf:.3f}")

        # Аннотированное изображение
        _draw_boxes(image, pred_boxes, (0, 255, 0), "cup")
        _draw_boxes(image, [(b[0], b[1], b[2], b[3]) for b in
                            [(gt[0], gt[1], gt[2], gt[3], 1.0) for gt in gt_boxes]],
                    (0, 165, 255), "gt")
        out_path = output_folder / f"annotated_{img_path.name}"
        try:
            cv2.imwrite(str(out_path), image)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Не удалось сохранить {out_path}: {exc}[/yellow]")

    console.print(table)

    precision, recall, map50 = _compute_map(predictions, ground_truths)
    summary = Table(title="Сводка по baseline")
    summary.add_column("Метрика")
    summary.add_column("Значение", justify="right")
    summary.add_row("Изображений", str(len(images)))
    summary.add_row("Всего детекций", str(total_dets))
    summary.add_row("Средняя уверенность",
                    f"{(total_conf_sum / total_dets):.3f}" if total_dets else "—")
    summary.add_row("Precision", f"{precision:.3f}")
    summary.add_row("Recall", f"{recall:.3f}")
    summary.add_row("mAP@0.5 (упрощ.)", f"{map50:.3f}")
    console.print(summary)

    console.print(
        f"\n[green]Аннотированные изображения сохранены в: {output_folder}[/green]\n"
        f"[cyan]Рекомендация:[/cyan] если mAP@0.5 < 0.85 или модель путает стакан "
        f"с кружкой — выполните fine-tune через finetune.py."
    )


@click.command()
@click.option("--images", "-i", "images_dir", required=True, type=click.Path(exists=True),
              help="Папка с тестовыми изображениями.")
@click.option("--class-id", "-c", default=DEFAULT_CLASS_ID, type=int,
              help=f"ID класса COCO (по умолчанию {DEFAULT_CLASS_ID} — cup).")
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Имя/путь к YOLOv8 модели.")
@click.option("--output", "-o", "output_dir", default="annotated_baseline",
              type=click.Path(), help="Куда сохранять аннотированные изображения.")
@click.option("--conf", default=CONF_THRESHOLD, type=float, help="Порог уверенности.")
@click.option("--iou", default=IOU_THRESHOLD, type=float, help="Порог IoU для NMS.")
def main(images_dir: str, class_id: int, model: str, output_dir: str,
         conf: float, iou: float) -> None:
    """Тест готовой YOLOv8 (COCO) на тестовых изображениях бумажных стаканов."""
    console.print(f"[bold cyan]COFFEE detection — baseline test[/bold cyan]")
    console.print(f"Папка изображений: {images_dir}")
    console.print(f"Модель: {model}, класс COCO id={class_id} (cup)")
    console.print(f"Порог conf={conf}, IoU={iou}")
    console.print(f"Выходная папка: {output_dir}\n")
    try:
        run_baseline(
            images_folder=Path(images_dir),
            output_folder=Path(output_dir),
            model_name=model,
            class_id=class_id,
            conf=conf,
            iou=iou,
        )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Непредвиденная ошибка: {exc}[/red]")
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
