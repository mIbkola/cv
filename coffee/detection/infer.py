#!/usr/bin/env python3
"""infer.py — инференс YOLOv8 + 3D-локализация через RealSense D435.

Получает кадр RGB+depth, находит стакан, возвращает 3D-координаты центра
стакана в системе координат камеры (X — вправо, Y — вниз, Z — от камеры).

Если RealSense не подключён или ``pyrealsense2`` не установлен — работает
в mock-режиме: читает кадры из указанной папки (RGB+depth пары). В логах
печатается ``[MOCK]``.

Возвращаемые данные (stdout, JSON):
    {
        "found": true,
        "class_name": "cup",
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.87,
        "xyz_camera": [x, y, z],   # метры
        "source": "realsense" | "mock"
    }

Пример запуска::

    # С реальной камерой:
    python infer.py --model best.pt --class-id 0 --live

    # Mock-режим (из папки):
    python infer.py --model yolov8n.pt --class-id 41 --mock-dir frames/
"""

from __future__ import annotations

# --- Импорты ---
import json
import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

# --- Константы ---
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CLASS_ID = 41  # COCO "cup" для baseline; 0 — для fine-tuned
CONF_THRESHOLD = 0.30
DEPTH_MIN_M = 0.2  # RealSense D435: минимальная дистанция ~0.2 м
DEPTH_MAX_M = 3.0  # обрезаем дальние шумы
RGB_WIN = "coffee_infer_rgb"

console = Console()


# --- Импорты сторонних библиотек ---
def _import_ultralytics():
    """Ленивый импорт ultralytics."""
    try:
        from ultralytics import YOLO  # type: ignore
        return YOLO
    except ImportError as exc:  # pragma: no cover
        console.print(
            "[red]Ошибка: ultralytics не установлен.[/red]\n"
            "Установите: [cyan]pip install ultralytics>=8.0.0[/cyan]"
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
    """Ленивый импорт pyrealsense2. Возвращает None если не установлен."""
    try:
        import pyrealsense2 as rs  # type: ignore
        return rs
    except ImportError:
        return None


# --- RealSense ---
class RealSenseCamera:
    """Обёртка над pyrealsense2 pipeline."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        rs = _import_realsense()
        if rs is None:
            raise RuntimeError("pyrealsense2 не установлен")
        self.rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        # intrinsics достанем после старта
        self._intrinsics = None
        try:
            self.pipeline.start(cfg)
            profile = self.pipeline.get_active_profile()
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self._intrinsics = color_stream.get_intrinsics()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Не удалось запустить RealSense: {exc}") from exc

    def get_frames(self) -> tuple[Optional[object], Optional[object]]:
        """Возвращает (color_image, depth_image_meters) или (None, None) по таймауту."""
        rs = self.rs
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Ошибка получения кадров RealSense: {exc}[/red]")
            return None, None
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None
        color = np_from_rs(color_frame)
        depth = np_from_rs(depth_frame).astype("float32") * 0.001  # мм → м
        return color, depth

    @property
    def intrinsics(self):
        return self._intrinsics

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:  # noqa: BLE001
            pass


def np_from_rs(frame):
    """Конвертация RealSense-фрейма в numpy array."""
    import numpy as np  # noqa
    return np.asanyarray(frame.get_data())


# --- Mock-режим ---
class MockFrames:
    """Читает кадры RGB + depth из папки.

    Имена файлов:
        <name>.jpg / .png — RGB
        <name>_depth.png / .npy — depth (метры, float32)
    """

    def __init__(self, folder: Path):
        self.folder = folder
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.rgb_files = sorted(p for p in folder.glob("*") if p.suffix.lower() in exts
                                and "_depth" not in p.stem)
        if not self.rgb_files:
            raise RuntimeError(f"В папке {folder} нет изображений для mock-режима")
        self._idx = 0

    def get_frames(self) -> tuple[Optional[object], Optional[object]]:
        cv2 = _import_cv2()
        import numpy as np  # noqa
        if self._idx >= len(self.rgb_files):
            return None, None
        rgb_path = self.rgb_files[self._idx]
        color = cv2.imread(str(rgb_path))
        if color is None:
            return None, None
        # Ищем depth: <stem>_depth.png или .npy рядом
        depth_png = rgb_path.with_name(f"{rgb_path.stem}_depth.png")
        depth_npy = rgb_path.with_name(f"{rgb_path.stem}_depth.npy")
        if depth_npy.exists():
            depth = np.load(depth_npy).astype("float32")
        elif depth_png.exists():
            d = cv2.imread(str(depth_png), cv2.IMREAD_ANYDEPTH)
            depth = (d.astype("float32") / 1000.0) if d is not None else None
        else:
            # Симулированная depth: бьём по центру кадра 1.0 м
            h, w = color.shape[:2]
            depth = np.full((h, w), 1.0, dtype="float32")
            console.print(f"[yellow][MOCK] depth для {rgb_path.name} сгенерирована "
                          f"симулированно (1.0 м).[/yellow]")
        self._idx += 1
        return color, depth


# --- 3D-локализация ---
def deproject_pixel_to_point(intrinsics, u: int, v: int, depth_m: float) -> tuple[float, float, float]:
    """Депроекция пикселя в 3D-точку. Использует pyrealsense2 если есть."""
    rs = _import_realsense2_safe()
    if rs is not None and intrinsics is not None:
        try:
            pt = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth_m)
            return float(pt[0]), float(pt[1]), float(pt[2])
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]rs2_deproject_pixel_to_point не сработал: {exc}[/yellow]")
    # Фолбэк: простая пинхол-модель
    fx = fy = 600.0
    cx = 320.0
    cy = 240.0
    if intrinsics is not None:
        fx = getattr(intrinsics, "fx", fx)
        fy = getattr(intrinsics, "fy", fy)
        cx = getattr(intrinsics, "ppx", cx)
        cy = getattr(intrinsics, "ppy", cy)
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    return x, y, depth_m


def _import_realsense2_safe():
    return _import_realsense()


def bbox_center_depth(depth, x1: int, y1: int, x2: int, y2: int) -> Optional[float]:
    """Берёт медианную глубину по центральной ROI bbox'а (устойчиво к краям стакана)."""
    import numpy as np  # noqa
    h, w = depth.shape
    # Центральная область 50% по каждой стороне
    cx1 = int(x1 + (x2 - x1) * 0.25)
    cx2 = int(x1 + (x2 - x1) * 0.75)
    cy1 = int(y1 + (y2 - y1) * 0.25)
    cy2 = int(y1 + (y2 - y1) * 0.75)
    cx1, cx2 = max(0, cx1), min(w, cx2)
    cy1, cy2 = max(0, cy1), min(h, cy2)
    roi = depth[cy1:cy2, cx1:cx2]
    valid = roi[(roi > DEPTH_MIN_M) & (roi < DEPTH_MAX_M)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def detect_and_localize(
    model,
    color,
    depth,
    class_id: int,
    conf: float,
    intrinsics,
    source: str,
):
    """Детекция стакана + 3D-локализация. Возвращает dict-результат."""
    cv2 = _import_cv2()
    results = model.predict(source=color, conf=conf, classes=[class_id], verbose=False)
    best = None
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for b in r.boxes:
            conf_b = float(b.conf[0])
            if best is None or conf_b > best[1]:
                best = (b.xyxy[0].tolist(), conf_b)
    if best is None:
        return {"found": False, "class_name": "cup", "source": source}

    xyxy, conf_b = best
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    cx_px = int((x1 + x2) / 2)
    cy_px = int((y1 + y2) / 2)
    depth_m = bbox_center_depth(depth, int(x1), int(y1), int(x2), int(y2))
    if depth_m is None:
        console.print("[yellow]Не удалось получить глубину для bbox (вне диапазона 0.2–3.0 м).[/yellow]")
        return {
            "found": True, "class_name": "cup",
            "bbox": [x1, y1, x2, y2], "confidence": conf_b,
            "xyz_camera": None, "source": source,
        }
    x_m, y_m, z_m = deproject_pixel_to_point(intrinsics, cx_px, cy_px, depth_m)

    # Визуализация (не падает если нет дисплея)
    try:
        vis = color.copy()
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(vis, (cx_px, cy_px), 4, (0, 0, 255), -1)
        text = f"({x_m:.2f},{y_m:.2f},{z_m:.2f}) m  conf={conf_b:.2f}"
        cv2.putText(vis, text, (int(x1), max(15, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("coffee_infer_rgb", vis)
        cv2.waitKey(1)
    except Exception:  # noqa: BLE001
        pass

    return {
        "found": True,
        "class_name": "cup",
        "bbox": [x1, y1, x2, y2],
        "confidence": conf_b,
        "xyz_camera": [x_m, y_m, z_m],
        "source": source,
    }


def run_live(model, class_id: int, conf: float, max_frames: int) -> list[dict]:
    """Запуск с живой RealSense D435 камерой."""
    rs = _import_realsense2_safe()
    if rs is None:
        console.print(
            "[yellow]pyrealsense2 не установлен → переключение в mock-режим невозможно "
            "без --mock-dir. Установите [cyan]pip install pyrealsense2[/cyan].[/yellow]"
        )
        raise SystemExit(2)
    try:
        cam = RealSenseCamera()
    except RuntimeError as exc:
        console.print(f"[yellow]RealSense не подключён: {exc}[/yellow]")
        console.print("[yellow]Переключаюсь в mock-режим, если указан --mock-dir.[/yellow]")
        raise
    results: list[dict] = []
    try:
        for i in range(max_frames):
            color, depth = cam.get_frames()
            if color is None or depth is None:
                console.print(f"[yellow]Кадр {i}: пустой RealSense-фрейм, пропуск.[/yellow]")
                continue
            res = detect_and_localize(
                model, color, depth, class_id, conf, cam.intrinsics, source="realsense"
            )
            console.print(f"[Кадр {i}] {res}")
            results.append(res)
            time.sleep(0.05)
    finally:
        cam.close()
    return results


def run_mock(model, class_id: int, conf: float, folder: Path) -> list[dict]:
    """Запуск в mock-режиме из папки с кадрами."""
    mock = MockFrames(folder)
    results: list[dict] = []
    while True:
        color, depth = mock.get_frames()
        if color is None:
            break
        # Intrinsics нет — fallback внутри deproject_pixel_to_point
        res = detect_and_localize(
            model, color, depth, class_id, conf, intrinsics=None, source="mock"
        )
        console.print(f"[MOCK] {res}")
        results.append(res)
    return results


@click.command()
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Путь/имя модели YOLOv8.")
@click.option("--class-id", "-c", default=DEFAULT_CLASS_ID, type=int,
              help="ID класса (41=COCO cup, 0=после fine-tune).")
@click.option("--conf", default=CONF_THRESHOLD, type=float, help="Порог уверенности.")
@click.option("--live", is_flag=True, default=False, help="Использовать живую RealSense камеру.")
@click.option("--mock-dir", default=None, type=click.Path(),
              help="Папка с кадрами для mock-режима (RGB + _depth.png/npy).")
@click.option("--max-frames", default=1, type=int,
              help="Сколько кадров обработать в live-режиме.")
@click.option("--output", "-o", "output_json", default=None, type=click.Path(),
              help="Куда сохранить результаты в JSON (иначе только печать).")
def main(model: str, class_id: int, conf: float, live: bool, mock_dir: str | None,
         max_frames: int, output_json: str | None) -> None:
    """Инференс YOLOv8 + 3D-локализация стакана через RealSense D435."""
    console.print("[bold cyan]COFFEE detection — infer + 3D localisation[/bold cyan]")
    YOLO = _import_ultralytics()
    try:
        yolo = YOLO(model)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Не удалось загрузить модель {model}: {exc}[/red]")
        raise SystemExit(3)

    if not live and not mock_dir:
        console.print(
            "[yellow]Не указан режим работы. Используйте --live или --mock-dir <папка>.[/yellow]"
        )
        raise SystemExit(2)

    try:
        if live:
            results = run_live(yolo, class_id, conf, max_frames)
        else:
            results = run_mock(yolo, class_id, conf, Path(mock_dir))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Непредвиденная ошибка: {exc}[/red]")
        console.print_exception()
        sys.exit(1)

    # Печать итогового (последнего) результата в JSON в stdout
    if results:
        print(json.dumps(results[-1], ensure_ascii=False))
    if output_json and results:
        Path(output_json).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        console.print(f"[green]Результаты сохранены в {output_json}[/green]")


if __name__ == "__main__":
    main()
