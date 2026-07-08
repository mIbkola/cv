#!/usr/bin/env python3
"""face_id.py — распознавание Олега по лицу.

Эталонное фото Олега лежит в ``data/oleg_face.jpg``. Скрипт:
1. Загружает эталон, считает face embedding через ``face_recognition``.
2. Получает кадр (с RealSense или из mock-папки).
3. Детектирует лица, сравнивает с эталоном (tolerance по умолчанию 0.6).
4. Возвращает True/False + 3D-позицию лица Олега в системе камеры.

Возвращает JSON:
    {
        "found": true,
        "name": "Oleg",
        "confidence": 0.93,
        "bbox": [...],
        "xyz_camera": [x, y, z],
        "source": "realsense" | "mock"
    }

Если ``face_recognition`` не установлен — печатается понятная ошибка с
подсказкой установки. Если RealSense недоступен — работает mock-режим
(читает кадры из папки, depth симулируется).

Пример::

    python face_id.py --reference data/oleg_face.jpg --live --max-frames 5
    python face_id.py --reference data/oleg_face.jpg --mock-dir frames/
"""

from __future__ import annotations

# --- Импорты ---
import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

# --- Константы ---
DEFAULT_TOLERANCE = 0.6  # Чем меньше — тем строже (0.6 — стандарт face_recognition)
DEFAULT_MODEL_ENC = "hog"  # "hog" (CPU) или "cnn" (GPU, точнее, медленнее)
DEFAULT_MAX_FRAMES = 5
DEFAULT_DEPTH_M = 1.0  # Глубина по умолчанию для mock-режима

console = Console()


def _import_face_recognition():
    """Ленивый импорт face_recognition с понятной ошибкой."""
    try:
        import face_recognition  # type: ignore
        return face_recognition
    except ImportError as exc:  # pragma: no cover
        console.print(
            "[red]Ошибка: библиотека face_recognition не установлена.[/red]\n"
            "Установите: [cyan]pip install face_recognition[/cyan]\n"
            "(нужен dlib, см. инструкции по сборке dlib для вашей ОС)."
        )
        raise SystemExit(2) from exc


def _import_cv2():
    """Ленивый импорт OpenCV."""
    try:
        import cv2  # type: ignore
        return cv2
    except ImportError as exc:  # pragma: no cover
        console.print(
            "[red]Ошибка: opencv-python не установлен.[/red]\n"
            "Установите: [cyan]pip install opencv-python[/cyan]"
        )
        raise SystemExit(2) from exc


def _import_realsense():
    """Ленивый импорт pyrealsense2. None если не установлен."""
    try:
        import pyrealsense2 as rs  # type: ignore
        return rs
    except ImportError:
        return None


# --- Эталонное лицо ---
def load_reference_encoding(reference_path: Path, model: str) -> list:
    """Считает face embedding с эталонного фото Олега."""
    fr = _import_face_recognition()
    if not reference_path.exists():
        console.print(f"[red]Ошибка: эталонное фото не найдено: {reference_path}[/red]")
        raise SystemExit(2)
    image = fr.load_image_file(str(reference_path))
    encodings = fr.face_encodings(image, model=model)
    if not encodings:
        console.print(
            f"[red]На эталонном фото {reference_path} не найдено лицо.[/red]\n"
            "Используйте чёткое фото анфас при хорошем освещении."
        )
        raise SystemExit(3)
    if len(encodings) > 1:
        console.print(
            f"[yellow]На эталонном фото найдено {len(encodings)} лиц. "
            f"Берётся первое.[/yellow]"
        )
    return encodings[0]


# --- 3D-локализация лица ---
def deproject_face_center(
    depth,
    bbox: tuple[int, int, int, int],
    intrinsics,
) -> Optional[tuple[float, float, float]]:
    """Депроектирует центр bbox лица в 3D (метры)."""
    import numpy as np  # noqa
    x1, y1, x2, y2 = bbox
    cx_px = int((x1 + x2) / 2)
    cy_px = int((y1 + y2) / 2)
    h, w = depth.shape
    # ROI вокруг центра лица
    rx1 = max(0, cx_px - 10)
    rx2 = min(w, cx_px + 10)
    ry1 = max(0, cy_px - 10)
    ry2 = min(h, cy_px + 10)
    roi = depth[ry1:ry2, rx1:rx2]
    valid = roi[(roi > 0.2) & (roi < 5.0)]
    if valid.size == 0:
        return None
    depth_m = float(np.median(valid))

    rs = _import_realsense()
    if rs is not None and intrinsics is not None:
        try:
            pt = rs.rs2_deproject_pixel_to_point(intrinsics, [cx_px, cy_px], depth_m)
            return float(pt[0]), float(pt[1]), float(pt[2])
        except Exception:  # noqa: BLE001
            pass
    # Пинхол-фолбэк
    fx = fy = 600.0
    cx = 320.0
    cy = 240.0
    if intrinsics is not None:
        fx = getattr(intrinsics, "fx", fx)
        fy = getattr(intrinsics, "fy", fy)
        cx = getattr(intrinsics, "ppx", cx)
        cy = getattr(intrinsics, "ppy", cy)
    x = (cx_px - cx) * depth_m / fx
    y = (cy_px - cy) * depth_m / fy
    return x, y, depth_m


# --- RealSense ---
def make_realsense_camera():
    """Создаёт обёртку над RealSense или возвращает None."""
    rs = _import_realsense()
    if rs is None:
        return None
    try:
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(cfg)
        profile = pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_stream.get_intrinsics()

        import numpy as np  # noqa

        def grab():
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                return None, None
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data()).astype("float32") * 0.001
            return color, depth

        def stop():
            try:
                pipeline.stop()
            except Exception:  # noqa: BLE001
                pass

        return grab, stop, intrinsics
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]RealSense недоступен: {exc}[/yellow]")
        return None


# --- Mock ---
def make_mock_source(folder: Path):
    """Возвращает генератор кадров из папки."""
    cv2 = _import_cv2()
    import numpy as np  # noqa
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted(p for p in folder.glob("*") if p.suffix.lower() in exts
                   and "_depth" not in p.stem)
    if not files:
        raise RuntimeError(f"В папке {folder} нет изображений")

    def grab():
        for p in files:
            color = cv2.imread(str(p))
            if color is None:
                continue
            h, w = color.shape[:2]
            depth = np.full((h, w), DEFAULT_DEPTH_M, dtype="float32")
            yield color, depth, p.name

    return grab()


# --- Детекция ---
def find_oleg_in_frame(
    fr,
    color,
    depth,
    reference_encoding,
    tolerance: float,
    model: str,
    intrinsics,
    source: str,
) -> list[dict]:
    """Ищет Олега на одном кадре. Возвращает список результатов."""
    cv2 = _import_cv2()
    # face_recognition работает с RGB
    import numpy as np  # noqa
    rgb = color[:, :, ::-1] if color.shape[2] == 3 else color
    face_locations = fr.face_locations(rgb, model=model)
    if not face_locations:
        return [{"found": False, "name": "Oleg", "source": source}]
    encodings = fr.face_encodings(rgb, face_locations)
    results: list[dict] = []
    for loc, enc in zip(face_locations, encodings):
        # loc = (top, right, bottom, left)
        top, right, bottom, left = loc
        distance = float(fr.face_distance([reference_encoding], enc)[0])
        match = bool(distance <= tolerance)
        confidence = max(0.0, 1.0 - distance)  # условная «уверенность»
        bbox = (left, top, right, bottom)
        xyz = deproject_face_center(depth, bbox, intrinsics) if depth is not None else None
        results.append({
            "found": match,
            "name": "Oleg" if match else "unknown",
            "confidence": round(confidence, 3),
            "distance": round(distance, 3),
            "bbox": list(bbox),
            "xyz_camera": list(xyz) if xyz else None,
            "source": source,
        })
        # Визуализация
        try:
            color_box = (0, 255, 0) if match else (0, 0, 255)
            label = f"Oleg {confidence:.2f}" if match else f"unknown {confidence:.2f}"
            cv2.rectangle(color, (left, top), (right, bottom), color_box, 2)
            cv2.putText(color, label, (left, max(15, top - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 1)
        except Exception:  # noqa: BLE001
            pass
    return results


@click.command()
@click.option("--reference", "-r", required=True, type=click.Path(exists=True),
              help="Эталонное фото Олега (data/oleg_face.jpg).")
@click.option("--live", is_flag=True, default=False, help="Live-режим с RealSense.")
@click.option("--mock-dir", default=None, type=click.Path(),
              help="Папка с кадрами для mock-режима.")
@click.option("--tolerance", default=DEFAULT_TOLERANCE, type=float,
              help="Порог расстояния face_recognition (меньше — строже).")
@click.option("--model", "enc_model", default=DEFAULT_MODEL_ENC,
              type=click.Choice(["hog", "cnn"]),
              help="Модель детекции лица: 'hog' (CPU) или 'cnn' (GPU).")
@click.option("--max-frames", default=DEFAULT_MAX_FRAMES, type=int,
              help="Сколько кадров обработать в live-режиме.")
@click.option("--output", "-o", "output_json", default=None, type=click.Path(),
              help="Куда сохранить JSON-результаты.")
def main(reference: str, live: bool, mock_dir: str | None, tolerance: float,
         enc_model: str, max_frames: int, output_json: str | None) -> None:
    """Распознавание Олега по лицу (face_recognition + RealSense D435)."""
    console.print("[bold cyan]COFFEE detection — face ID (Олег)[/bold cyan]")
    fr = _import_face_recognition()
    ref_enc = load_reference_encoding(Path(reference), enc_model)
    console.print(f"[green]Эталонное лицо загружено: {reference}[/green]")

    if not live and not mock_dir:
        console.print(
            "[yellow]Не указан режим. Используйте --live или --mock-dir <папка>.[/yellow]"
        )
        raise SystemExit(2)

    all_results: list[dict] = []

    try:
        if live:
            cam = make_realsense_camera()
            if cam is None:
                console.print(
                    "[yellow]RealSense недоступен. Укажите --mock-dir для mock-режима.[/yellow]"
                )
                raise SystemExit(2)
            grab, stop, intrinsics = cam
            try:
                for i in range(max_frames):
                    color, depth = grab()
                    if color is None or depth is None:
                        continue
                    res = find_oleg_in_frame(
                        fr, color, depth, ref_enc, tolerance, enc_model, intrinsics, "realsense"
                    )
                    console.print(f"[Кадр {i}] {res}")
                    all_results.extend(res)
            finally:
                stop()
        else:
            intrinsics = None
            for color, depth, name in make_mock_source(Path(mock_dir)):
                res = find_oleg_in_frame(
                    fr, color, depth, ref_enc, tolerance, enc_model, intrinsics, "mock"
                )
                for r in res:
                    r["file"] = name
                console.print(f"[MOCK] {res}")
                all_results.extend(res)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Непредвиденная ошибка: {exc}[/red]")
        console.print_exception()
        sys.exit(1)

    # Итог: хотя бы одно подтверждённое совпадение?
    found = any(r.get("found") for r in all_results)
    console.print(
        f"\n[bold {'green' if found else 'red'}]"
        f"Олег {'НАЙДЕН' if found else 'НЕ найден'} на кадрах.[/bold {'green' if found else 'red'}]"
    )

    if all_results:
        print(json.dumps(all_results[-1], ensure_ascii=False))
    if output_json and all_results:
        Path(output_json).write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        console.print(f"[green]Результаты сохранены в {output_json}[/green]")


if __name__ == "__main__":
    main()
