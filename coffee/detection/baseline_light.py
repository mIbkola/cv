#!/usr/bin/env python3
"""baseline_light.py — лёгкий CV-детектор бумажных стаканов на чистом OpenCV.

Детектор БЕЗ нейросетей и БЕЗ ultralytics. Использует только opencv-python,
numpy и (опционально) scikit-learn. Подходит для fallback-режима, когда
ultralytics установить нельзя (нет места под torch и т.п.).

Алгоритм:
    1. Конвертация BGR → HSV.
    2. Цветовая сегментация: объединяем 3 маски под типичные цвета стакана:
         - бежевый/кремовый  (H=20-40,  S=20-80,  V=150-255)
         - коричневый        (H=10-25,  S=50-150, V=80-200)
         - белый бумажный    (H=0-180,  S=0-30,   V=180-255)
    3. Морфологические операции (closing: dilate+erode) для устранения шума.
    4. findContours → фильтрация по площади (>500 px²) и aspect ratio (0.5-2.0).
    5. Эвристика: стакан обычно выше чем шире (h/w > 1.0).
    6. Confidence: на основе площади контура + плотности маски внутри bbox.
    7. NMS (non-max suppression) для подавления дубликатов.

CLI::

    python baseline_light.py --images test_images/ --visualize
    python baseline_light.py --images test_images/ --visualize --output annotated/
"""

from __future__ import annotations

# --- Импорты ---
import json
import sys
from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

# --- Константы ---
MIN_AREA = 500                # минимальная площадь контура, px²
MAX_AREA_FRAC = 0.45          # максимальная доля изображения (стакан не может быть > 45% кадра)
ASPECT_MIN = 0.4              # минимальное w/h (стакан может быть узким)
ASPECT_MAX = 2.0              # максимальное w/h
HEIGHT_RATIO_MIN = 0.6        # эвристика: h/w должно быть >= 0.6 (стакан вытянут вверх).
                              # (не строго >1.0 — после sleeve-split часть может быть ~0.6)
IOU_EVAL_THR = 0.3            # порог IoU для метрик precision/recall
NMS_IOU_THR = 0.25            # порог IoU для NMS внутри детектора (агрессивный)
MIN_CONF = 0.10               # минимальная confidence для выдачи
BG_MATCH_RATIO = 0.40         # если цветовая маска покрывает >40% кадра → фон совпал
                              # со стаканом по цвету → переключаемся на edge-based fallback.
MERGE_X_OVERLAP = 0.5         # порог перекрытия по X для слияния вертикальных фрагментов
MERGE_Y_GAP = 80              # максимальный зазор по Y для слияния (px)

# Цветовые диапазоны в HSV (OpenCV: H = 0-179, S = 0-255, V = 0-255)
# Подобраны под палитру генератора (бежевый/коричневый имеют H≈15-19).
# Каждый диапазон: (lower, upper)
COLOR_RANGES = [
    # бежевый / кремовый (базовый цвет бумажного стакана)
    ((12, 20, 140), (35, 110, 255)),
    # коричневый (тёмные оттенки, «кофейные» стаканы)
    ((8, 50, 70), (25, 180, 210)),
    # белый бумажный (любой оттенок, насыщенность низкая, яркость высокая)
    ((0, 0, 180), (180, 35, 255)),
]

DEFAULT_OUTPUT = "annotated"
DEFAULT_GT_NAME = "bounding_boxes.json"

console = Console()


# --- Утилиты ---

def _collect_images(folder: Path) -> list[Path]:
    """Собирает список изображений в папке (без рекурсии в подкаталоги)."""
    if not folder.exists():
        console.print(f"[red]Ошибка: папка не найдена: {folder}[/red]")
        raise SystemExit(2)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(p for p in folder.iterdir()
                    if p.suffix.lower() in exts and p.is_file())
    return images


def _iou_xyxy(a: tuple[float, float, float, float],
              b: tuple[float, float, float, float]) -> float:
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


def _nms(boxes: list[tuple[float, float, float, float, float]],
         iou_thr: float = NMS_IOU_THR) -> list[tuple[float, float, float, float, float]]:
    """Non-max suppression. boxes = [(x1, y1, x2, y2, conf), ...]."""
    if not boxes:
        return []
    boxes_sorted = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept: list[tuple[float, float, float, float, float]] = []
    while boxes_sorted:
        best = boxes_sorted.pop(0)
        kept.append(best)
        boxes_sorted = [b for b in boxes_sorted
                        if _iou_xyxy(b[:4], best[:4]) < iou_thr]
    return kept


# --- Детекция ---

def _build_color_mask(hsv: np.ndarray) -> np.ndarray:
    """Строит бинарную маску пикселей «под цвет бумажного стакана»."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in COLOR_RANGES:
        m = cv2.inRange(hsv, np.array(lower, dtype=np.uint8),
                        np.array(upper, dtype=np.uint8))
        mask = cv2.bitwise_or(mask, m)
    return mask


def _morphology(mask: np.ndarray) -> np.ndarray:
    """Морфологическая очистка маски (closing + opening).

    Closing с ядром 11x11 (1 итерация) заполняет мелкие дыры внутри стакана,
    но НЕ объединяет соседние стаканы (для этого есть _merge_vertical_fragments).
    """
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Closing: dilate → erode (заполняет дыры внутри стакана)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    # Opening: erode → dilate (убирает мелкий шум)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    return mask


def _border_touch_count(x: int, y: int, w: int, h: int,
                        img_w: int, img_h: int, margin: int = 2) -> int:
    """Сколько границ изображения касается bbox (0-4).

    Возвращает количество «задетых» границ. Используется для отсеивания
    «фоновых» компонентов — они обычно занимают весь кадр и касаются 3+ границ.
    Один край может касаться легитимный стакан, обрезанный краем кадра.
    """
    n = 0
    if x <= margin:
        n += 1
    if y <= margin:
        n += 1
    if x + w >= img_w - margin:
        n += 1
    if y + h >= img_h - margin:
        n += 1
    return n


def _touches_border(x: int, y: int, w: int, h: int,
                    img_w: int, img_h: int, margin: int = 2) -> bool:
    """True если bbox касается 3+ границ → почти наверняка это фон."""
    return _border_touch_count(x, y, w, h, img_w, img_h, margin) >= 3


def _contour_to_box(cnt: np.ndarray, img_w: int, img_h: int
                    ) -> Optional[tuple[int, int, int, int]]:
    """Пре-фильтр контура: только по площади и границам.

    Полная проверка формы (aspect ratio, h/w) делается ПОСЛЕ
    _merge_vertical_fragments — иначе фрагменты стакана, разрезанные
    sleeve'ом, отбрасываются до слияния.
    """
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    if area < MIN_AREA:
        return None
    # Слишком большие компоненты — это фон, а не стакан
    if area > img_w * img_h * MAX_AREA_FRAC:
        return None
    # Касается 3+ границ → скорее всего фон
    if _touches_border(x, y, w, h, img_w, img_h):
        return None
    return (x, y, w, h)


def _compute_confidence(mask: np.ndarray, x: int, y: int,
                        w: int, h: int, img_area: int) -> float:
    """Confidence = sqrt(площадь_bbox/img_area) * плотность_маски_в_bbox.

    Плотность маски — какая доля пикселей в bbox действительно попала в
    цветовой диапазон. Оба множителя в [0, 1].
    """
    H, W = mask.shape
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = mask[y1:y2, x1:x2]
    density = float(np.count_nonzero(roi)) / float((x2 - x1) * (y2 - y1))
    area_norm = max(0.0, min(1.0, (w * h) / max(1.0, img_area * 0.25)))
    # sqrt чтобы маленькие/больные bbox'ы не доминировали
    conf = float(np.sqrt(area_norm) * density)
    return round(conf, 3)


def _merge_vertical_fragments(
    boxes: list[tuple[int, int, int, int]],
    x_overlap_thr: float = MERGE_X_OVERLAP,
    y_gap_max: int = MERGE_Y_GAP,
) -> list[tuple[int, int, int, int]]:
    """Объединяет фрагменты стакана, разрезанные sleeve'ом.

    Если два bbox'а перекрываются по X больше чем на x_overlap_thr и зазор
    по Y меньше y_gap_max — это, скорее всего, верхняя и нижняя части одного
    стакана (разделённые тёмным sleeve). Объединяем в один.
    """
    if len(boxes) <= 1:
        return list(boxes)
    # Сортируем по Y (сверху вниз)
    sorted_boxes = sorted(boxes, key=lambda b: b[1])
    merged: list[tuple[int, int, int, int]] = []
    for box in sorted_boxes:
        x, y, w, h = box
        if not merged:
            merged.append(box)
            continue
        # Проверяем слияние с последним в merged
        last = merged[-1]
        lx, ly, lw, lh = last
        # Перекрытие по X
        x_overlap = max(0, min(x + w, lx + lw) - max(x, lx))
        min_w = min(w, lw)
        if min_w > 0 and x_overlap / min_w >= x_overlap_thr:
            # Зазор по Y
            y_gap = y - (ly + lh)
            if 0 <= y_gap <= y_gap_max:
                # Объединяем
                new_x = min(lx, x)
                new_y = min(ly, y)
                new_x2 = max(lx + lw, x + w)
                new_y2 = max(ly + lh, y + h)
                merged[-1] = (new_x, new_y, new_x2 - new_x, new_y2 - new_y)
                continue
        merged.append(box)
    return merged


def _detect_via_edges(image_bgr: np.ndarray, color_mask: np.ndarray
                      ) -> list[tuple[int, int, int, int]]:
    """Edge-based fallback когда фон совпадает со стаканом по цвету.

    Использует Canny + дилатацию для поиска замкнутых контуров. Затем
    фильтрует по форме и проверяет, что внутри контура достаточно
    «стаканных» пикселей.

    Использует более строгие фильтры, чем color-режим:
    - минимальная площадь 1500 px² (вместо 500)
    - aspect ratio 0.5-1.5 (ближе к форме стакана)
    - h/w >= 0.8 (стакан явно вытянут вверх)
    - color_density внутри bbox >= 0.55
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # Адаптивные пороги Canny по медиане
    v = float(np.median(gray))
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(gray, lower, upper)
    # Дилатация чтобы закрыть мелкие разрывы в контуре
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    # Closing чтобы объединить близкие фрагменты в один контур
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_k, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Строгие пороги для edge-режима (баланс precision/recall)
    EDGE_MIN_AREA = 1000
    EDGE_ASPECT_MIN = 0.5
    EDGE_ASPECT_MAX = 1.5
    EDGE_HW_MIN = 0.8
    EDGE_COLOR_DENSITY_MIN = 0.50

    candidates: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < EDGE_MIN_AREA:
            continue
        if area > h * w * MAX_AREA_FRAC:
            continue
        if _touches_border(x, y, bw, bh, w, h):
            continue
        aspect = bw / bh if bh > 0 else 0
        if not (EDGE_ASPECT_MIN <= aspect <= EDGE_ASPECT_MAX):
            continue
        if (bh / bw if bw > 0 else 0) < EDGE_HW_MIN:
            continue
        # Проверяем «стаканные» пиксели внутри bbox'а
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        if x2 <= x1 or y2 <= y1:
            continue
        roi_color = color_mask[y1:y2, x1:x2]
        color_density = float(np.count_nonzero(roi_color)) / \
            float((x2 - x1) * (y2 - y1))
        if color_density < EDGE_COLOR_DENSITY_MIN:
            continue
        candidates.append((x, y, bw, bh))
    return candidates


def detect_cups(image_bgr: np.ndarray) -> list[dict]:
    """Детектирует бумажные стаканы на изображении BGR.

    Возвращает список dict'ов: [{x, y, w, h, confidence}, ...].
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    h, w = image_bgr.shape[:2]
    img_area = h * w

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    color_mask = _build_color_mask(hsv)
    mask = _morphology(color_mask)

    # Доля изображения, покрытая маской
    mask_ratio = float(np.count_nonzero(mask)) / float(mask.size)

    raw_bboxes: list[tuple[int, int, int, int]] = []

    if mask_ratio > BG_MATCH_RATIO:
        # Фон совпал со стаканом по цвету — переключаемся на edge-based fallback
        edge_boxes = _detect_via_edges(image_bgr, color_mask)
        raw_bboxes.extend(edge_boxes)
    else:
        # Цветовая сегментация работает нормально
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            bbox = _contour_to_box(cnt, w, h)
            if bbox is None:
                continue
            raw_bboxes.append(bbox)
        # Объединяем фрагменты, разрезанные sleeve'ом
        raw_bboxes = _merge_vertical_fragments(raw_bboxes)
        # Повторно фильтруем после слияния (могли стать слишком большими)
        raw_bboxes = [b for b in raw_bboxes
                      if _passes_final_filter(b, w, h)]

    # Считаем confidence для каждого
    raw_boxes: list[tuple[float, float, float, float, float]] = []
    for (x, y, bw, bh) in raw_bboxes:
        conf = _compute_confidence(color_mask, x, y, bw, bh, img_area)
        # Если confidence по цвету низкая (фон совпал) — используем площадь как fallback,
        # но с понижающим коэффициентом (edge-режим менее надёжен)
        if conf < MIN_CONF:
            area_norm = max(0.0, min(1.0, (bw * bh) / max(1.0, img_area * 0.10)))
            conf = round(float(np.sqrt(area_norm) * 0.35), 3)
        if conf < MIN_CONF:
            continue
        raw_boxes.append((float(x), float(y), float(x + bw), float(y + bh), conf))

    # NMS для подавления дубликатов
    kept = _nms(raw_boxes, NMS_IOU_THR)

    results: list[dict] = []
    for (x1, y1, x2, y2, conf) in kept:
        results.append({
            "x": int(x1),
            "y": int(y1),
            "w": int(x2 - x1),
            "h": int(y2 - y1),
            "confidence": float(conf),
        })
    return results


def _passes_final_filter(box: tuple[int, int, int, int],
                         img_w: int, img_h: int) -> bool:
    """Финальная проверка bbox'а после слияния фрагментов."""
    x, y, w, h = box
    area = w * h
    if area < MIN_AREA:
        return False
    if area > img_w * img_h * MAX_AREA_FRAC:
        return False
    if _touches_border(x, y, w, h, img_w, img_h):
        return False
    aspect = w / h if h > 0 else 0
    if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
        return False
    if (h / w if w > 0 else 0) < HEIGHT_RATIO_MIN:
        return False
    return True


# --- Визуализация ---

def _draw_results(image_bgr: np.ndarray, detections: list[dict],
                  gt_boxes: list[dict] | None = None) -> np.ndarray:
    """Рисует bbox'ы детектора (зелёный) и опционально GT (оранжевый)."""
    out = image_bgr.copy()
    for d in detections:
        x, y, w, h = d["x"], d["y"], d["w"], d["h"]
        conf = d["confidence"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(out, f"cup {conf:.2f}", (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    if gt_boxes is not None:
        for g in gt_boxes:
            x, y, w, h = g["x"], g["y"], g["w"], g["h"]
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 165, 255), 1)
    return out


# --- Метрики ---

def _load_ground_truth(gt_path: Path) -> dict[str, list[dict]]:
    """Читает bounding_boxes.json и возвращает {image_name: [boxes]}."""
    if not gt_path.exists():
        return {}
    try:
        data = json.loads(gt_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]Не удалось прочитать GT {gt_path}: {exc}[/yellow]")
        return {}
    result: dict[str, list[dict]] = {}
    for entry in data.get("images", []):
        result[entry["image"]] = entry.get("boxes", [])
    return result


def _compute_metrics(
    predictions: dict[str, list[dict]],
    ground_truth: dict[str, list[dict]],
    iou_thr: float = IOU_EVAL_THR,
) -> tuple[float, float, int, int, int]:
    """Считает precision/recall по изображениям.

    Бокс считается TP, если IoU с каким-то GT > iou_thr (один GT → один match).
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for img_name, gts in ground_truth.items():
        preds = predictions.get(img_name, [])
        matched_gt = [False] * len(gts)
        # Сортируем предсказания по confidence (убывание) — greedy matching
        preds_sorted = sorted(preds, key=lambda p: p["confidence"], reverse=True)
        for p in preds_sorted:
            p_box = (p["x"], p["y"], p["x"] + p["w"], p["y"] + p["h"])
            best_iou, best_idx = 0.0, -1
            for j, g in enumerate(gts):
                if matched_gt[j]:
                    continue
                g_box = (g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"])
                iou = _iou_xyxy(p_box, g_box)
                if iou > best_iou:
                    best_iou, best_idx = iou, j
            if best_iou >= iou_thr and best_idx >= 0:
                matched_gt[best_idx] = True
                total_tp += 1
            else:
                total_fp += 1
        total_fn += sum(1 for m in matched_gt if not m)

    # Также считаем FP для предсказаний на изображениях, которых нет в GT
    for img_name, preds in predictions.items():
        if img_name in ground_truth:
            continue
        total_fp += len(preds)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    return precision, recall, total_tp, total_fp, total_fn


# --- Основная логика ---

def run(
    images_dir: Path,
    output_dir: Path,
    visualize: bool,
    gt_path: Path | None,
) -> dict:
    """Основная логика детектора. Возвращает summary dict."""
    images = _collect_images(images_dir)
    if not images:
        console.print(f"[yellow]В папке {images_dir} нет изображений.[/yellow]")
        return {"images": 0}

    if visualize:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем GT если есть
    gt_map: dict[str, list[dict]] = {}
    if gt_path and gt_path.exists():
        gt_map = _load_ground_truth(gt_path)
        console.print(f"[cyan]Загружен ground truth: {len(gt_map)} изображений "
                      f"из {gt_path.name}[/cyan]")
    elif gt_path is None:
        # Пробуем найти bounding_boxes.json в images_dir по умолчанию
        auto_gt = images_dir / DEFAULT_GT_NAME
        if auto_gt.exists():
            gt_map = _load_ground_truth(auto_gt)
            console.print(f"[cyan]Авто-найден ground truth: {auto_gt}[/cyan]")

    predictions: dict[str, list[dict]] = {}
    table = Table(title="baseline_light — детекция стаканов (OpenCV)")
    table.add_column("Файл", overflow="fold")
    table.add_column("Det", justify="right")
    table.add_column("GT", justify="right")
    table.add_column("Avg conf", justify="right")

    total_dets = 0
    total_conf_sum = 0.0

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            console.print(f"[yellow]Не удалось прочитать: {img_path}[/yellow]")
            continue

        dets = detect_cups(image)
        predictions[img_path.name] = dets

        gt_boxes = gt_map.get(img_path.name, [])
        avg_conf = (sum(d["confidence"] for d in dets) / len(dets)) if dets else 0.0
        total_dets += len(dets)
        total_conf_sum += sum(d["confidence"] for d in dets)
        table.add_row(img_path.name, str(len(dets)), str(len(gt_boxes)),
                      f"{avg_conf:.3f}")

        if visualize:
            annotated = _draw_results(image, dets, gt_boxes if gt_map else None)
            out_path = output_dir / f"annotated_{img_path.name}"
            try:
                cv2.imwrite(str(out_path), annotated)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Не удалось сохранить {out_path}: {exc}[/yellow]")

    # Печатаем таблицу только если изображений немного (иначе слишком длинная)
    if len(images) <= 20:
        console.print(table)
    else:
        # Печатаем первые 5 + последние 5 строк
        console.print(f"[cyan]Обработано {len(images)} изображений "
                      f"(таблица suppressed — слишком длинная).[/cyan]")

    summary = {
        "images": len(images),
        "total_dets": total_dets,
        "avg_conf": (total_conf_sum / total_dets) if total_dets else 0.0,
    }

    # Метрики если есть GT
    if gt_map:
        precision, recall, tp, fp, fn = _compute_metrics(predictions, gt_map)
        summary.update({
            "precision": precision,
            "recall": recall,
            "tp": tp, "fp": fp, "fn": fn,
            "total_gt": sum(len(v) for v in gt_map.values()),
        })

        m_table = Table(title="Метрики baseline_light (IoU >= 0.3)")
        m_table.add_column("Метрика")
        m_table.add_column("Значение", justify="right")
        m_table.add_row("Изображений", str(len(images)))
        m_table.add_row("Всего детекций", str(total_dets))
        m_table.add_row("Всего GT-боксов", str(summary["total_gt"]))
        m_table.add_row("TP", str(tp))
        m_table.add_row("FP", str(fp))
        m_table.add_row("FN", str(fn))
        m_table.add_row("Средняя confidence",
                        f"{summary['avg_conf']:.3f}" if total_dets else "—")
        m_table.add_row("Precision", f"{precision:.3f}")
        m_table.add_row("Recall", f"{recall:.3f}")
        m_table.add_row("F1",
                        f"{(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0:.3f}")
        console.print(m_table)
    else:
        s_table = Table(title="Сводка baseline_light")
        s_table.add_column("Метрика")
        s_table.add_column("Значение", justify="right")
        s_table.add_row("Изображений", str(len(images)))
        s_table.add_row("Всего детекций", str(total_dets))
        s_table.add_row("Средняя confidence",
                        f"{summary['avg_conf']:.3f}" if total_dets else "—")
        console.print(s_table)

    if visualize:
        console.print(f"\n[green]Аннотированные изображения сохранены в: "
                      f"{output_dir}[/green]")

    return summary


# --- CLI ---

@click.command()
@click.option("--images", "-i", "images_dir", required=True,
              type=click.Path(exists=True),
              help="Папка с тестовыми изображениями.")
@click.option("--visualize", is_flag=True, default=False,
              help="Сохранять аннотированные изображения.")
@click.option("--output", "-o", "output_dir", default=DEFAULT_OUTPUT,
              type=click.Path(),
              help=f"Куда сохранять аннотированные изображения "
                   f"(по умолчанию {DEFAULT_OUTPUT}).")
@click.option("--gt", "gt_path", default=None, type=click.Path(),
              help="Путь к bounding_boxes.json (если не указан — авто-поиск "
                   "в папке изображений).")
def main(images_dir: str, visualize: bool, output_dir: str,
         gt_path: str | None) -> None:
    """Лёгкий CV-детектор бумажных стаканов на чистом OpenCV (без YOLO)."""
    console.print("[bold cyan]COFFEE — baseline_light (OpenCV-only)[/bold cyan]")
    console.print(f"Папка изображений: {images_dir}")
    console.print(f"Визуализация: {'да' if visualize else 'нет'}")
    console.print(f"Выходная папка: {output_dir}\n")

    try:
        run(
            images_dir=Path(images_dir),
            output_dir=Path(output_dir),
            visualize=visualize,
            gt_path=Path(gt_path) if gt_path else None,
        )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Непредвиденная ошибка: {exc}[/red]")
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
