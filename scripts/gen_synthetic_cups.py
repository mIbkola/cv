#!/usr/bin/env python3
"""gen_synthetic_cups.py — генератор синтетических изображений бумажных стаканов.

Скрипт создаёт синтетические фотографии бумажных стаканов с кофе для
тестирования CV-детекторов без необходимости собирать реальный датасет.
Использует ТОЛЬКО Pillow + numpy (никаких тяжёлых зависимостей вроде torch
или ultralytics).

Что генерируется:
    - 50 (по умолчанию) изображений 640×480 пикселей
    - Случайный фон: однотонный / вертикальный градиент / текстура шума
      (имитация стола, кухонной столешницы, тёмной комнаты)
    - 1–3 стакана на каждом изображении:
        * Бумажный стакан (трапеция: верх шире низа)
        * Цвета: бежевый / кремовый / коричневый / белый
        * Опционально sleeve (картонная обёртка) — тёмная горизонтальная полоса
        * Опционально крышка (эллипс сверху)
        * Опционально «пар» (полупрозрачные белые волнистые линии)
    - Случайная позиция, лёгкий наклон (перспектива), масштаб
    - Ground truth JSON: bounding_boxes.json — для каждого изображения список
      bbox'ов {x, y, w, h, class}

Пример запуска::

    /home/z/.venv/bin/python3 scripts/gen_synthetic_cups.py --count 50
    /home/z/.venv/bin/python3 scripts/gen_synthetic_cups.py --count 100 \\
        --out coffee/detection/test_images/
"""

from __future__ import annotations

# --- Импорты ---
import json
import random
import sys
from pathlib import Path

import click
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# --- Константы ---
IMG_W = 640  # ширина изображения
IMG_H = 480  # высота изображения

# Палитра цветов фонов (имитация стола/кухни/столешницы)
BG_SOLID_COLORS = [
    (90, 60, 40),     # тёмное дерево
    (140, 100, 70),   # светлое дерево
    (60, 60, 65),     # тёмный камень
    (180, 175, 165),  # светлая столешница
    (70, 80, 90),     # серо-синяя кухня
    (200, 190, 170),  # кремовый стол
    (50, 50, 50),     # тёмный фон
    (165, 145, 120),  # песочный
]

# Палитра цветов стакана (RGB)
CUP_COLORS = [
    (220, 200, 170),   # бежевый (классический)
    (210, 185, 150),   # кремовый
    (180, 140, 90),    # коричневый
    (160, 120, 80),    # тёмный коричневый
    (245, 245, 240),   # белый бумажный
    (230, 215, 190),   # светло-бежевый
    (190, 165, 130),   # песочно-коричневый
]

# Палитра цветов sleeve (тёмная картонная обёртка)
SLEEVE_COLORS = [
    (50, 40, 30),      # почти чёрный
    (80, 55, 35),      # тёмно-коричневый
    (35, 35, 40),      # графит
    (110, 80, 50),     # средний коричневый
]

LID_COLORS = [
    (240, 240, 235),   # белый
    (220, 215, 205),   # кремовый
    (180, 180, 180),   # серый
    (250, 250, 248),   # чистый белый
]

DEFAULT_OUT = "coffee/detection/test_images"
DEFAULT_COUNT = 50

# --- Утилиты для фона ---

def _make_solid_background(color: tuple[int, int, int]) -> Image.Image:
    """Создаёт однотонный фон заданного цвета."""
    return Image.new("RGB", (IMG_W, IMG_H), color)


def _make_gradient_background(top: tuple[int, int, int],
                              bottom: tuple[int, int, int]) -> Image.Image:
    """Создаёт вертикальный градиент от top (сверху) к bottom (снизу)."""
    arr = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    for y in range(IMG_H):
        t = y / (IMG_H - 1)
        r = int(top[0] + (bottom[0] - top[0]) * t)
        g = int(top[1] + (bottom[1] - top[1]) * t)
        b = int(top[2] + (bottom[2] - top[2]) * t)
        arr[y, :] = (r, g, b)
    return Image.fromarray(arr)


def _make_noisy_background(base_color: tuple[int, int, int],
                           noise_strength: int = 25) -> Image.Image:
    """Создаёт однотонный фон с лёгким шумом (имитация текстуры стола)."""
    arr = np.full((IMG_H, IMG_W, 3), base_color, dtype=np.int16)
    noise = np.random.randint(
        -noise_strength, noise_strength + 1, size=(IMG_H, IMG_W, 1), dtype=np.int16
    )
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _make_random_background(rng: random.Random) -> Image.Image:
    """Выбирает случайный тип фона и возвращает готовое изображение."""
    bg_type = rng.choice(["solid", "gradient", "noisy", "noisy"])

    if bg_type == "solid":
        color = rng.choice(BG_SOLID_COLORS)
        return _make_solid_background(color)

    if bg_type == "gradient":
        top = rng.choice(BG_SOLID_COLORS)
        # нижний цвет немного темнее/светнее верхнего
        bottom = tuple(max(0, min(255, c + rng.randint(-40, 40))) for c in top)
        return _make_gradient_background(top, bottom)

    # noisy (имитация текстуры стола)
    color = rng.choice(BG_SOLID_COLORS)
    strength = rng.randint(15, 35)
    return _make_noisy_background(color, strength)


# --- Рисование стакана ---

def _draw_cup(
    img: Image.Image,
    rng: random.Random,
    center_x: int,
    bottom_y: int,
    cup_height: int,
    top_width: int,
    bottom_width: int,
    cup_color: tuple[int, int, int],
    skew_px: int,
    add_sleeve: bool,
    add_lid: bool,
    add_steam: bool,
) -> dict:
    """Рисует один бумажный стакан на изображении и возвращает его bbox.

    Возвращает dict {x, y, w, h, class} — bbox включает сам стакан и крышку
    (если есть), но НЕ включает пар (пар полупрозрачный и не входит в GT).
    """
    draw = ImageDraw.Draw(img, "RGBA")

    # Геометрия трапеции. Верх шире низа — типичная форма бумажного стакана.
    # skew_px — горизонтальное смещение верха относительно низа (лёгкая перспектива).
    top_left_x = center_x - top_width // 2 + skew_px
    top_right_x = center_x + top_width // 2 + skew_px
    bot_left_x = center_x - bottom_width // 2
    bot_right_x = center_x + bottom_width // 2
    top_y = bottom_y - cup_height

    # Точки трапеции (по часовой: TL, TR, BR, BL)
    polygon = [
        (top_left_x, top_y),
        (top_right_x, top_y),
        (bot_right_x, bottom_y),
        (bot_left_x, bottom_y),
    ]

    # Лёгкая вертикальная затенённость (правая сторона чуть темнее) — имитация объёма.
    base = cup_color
    darker = tuple(max(0, c - 25) for c in base)
    lighter = tuple(min(255, c + 15) for c in base)

    # Заливка основным цветом
    draw.polygon(polygon, fill=base)

    # Левая светлая полоса (блик)
    left_highlight = [
        (top_left_x, top_y),
        (top_left_x + max(2, top_width // 8), top_y),
        (bot_left_x + max(2, bottom_width // 8), bottom_y),
        (bot_left_x, bottom_y),
    ]
    draw.polygon(left_highlight, fill=lighter)

    # Правая затенённая полоса
    right_shade = [
        (top_right_x - max(2, top_width // 6), top_y),
        (top_right_x, top_y),
        (bot_right_x, bottom_y),
        (bot_right_x - max(2, bottom_width // 6), bottom_y),
    ]
    draw.polygon(right_shade, fill=darker)

    # Контур
    draw.line(polygon + [polygon[0]], fill=(0, 0, 0, 80), width=1)

    # --- Sleeve (картонная обёртка) ---
    if add_sleeve:
        sleeve_color = rng.choice(SLEEVE_COLORS)
        # Полоса занимает 25-40% высоты стакана, в средней части
        sleeve_h = max(15, int(cup_height * rng.uniform(0.25, 0.40)))
        # Y-диапазон sleeve внутри стакана
        s_top_frac = rng.uniform(0.30, 0.50)
        s_y1 = top_y + int(cup_height * s_top_frac)
        s_y2 = s_y1 + sleeve_h
        # Ширина стакана на уровне y1 и y2 (линейная интерполяция)
        def _width_at(y: int) -> int:
            t = (y - top_y) / max(1, cup_height)
            return int(top_width + (bottom_width - top_width) * t)

        w1 = _width_at(s_y1)
        w2 = _width_at(s_y2)
        # Учитываем skew
        cx1 = center_x + int(skew_px * (1 - (s_y1 - top_y) / max(1, cup_height)))
        cx2 = center_x + int(skew_px * (1 - (s_y2 - top_y) / max(1, cup_height)))

        sleeve_poly = [
            (cx1 - w1 // 2, s_y1),
            (cx1 + w1 // 2, s_y1),
            (cx2 + w2 // 2, s_y2),
            (cx2 - w2 // 2, s_y2),
        ]
        draw.polygon(sleeve_poly, fill=sleeve_color)
        # Лёгкий блик слева на sleeve
        sleeve_light = tuple(min(255, c + 30) for c in sleeve_color)
        sl_poly = [
            (cx1 - w1 // 2, s_y1),
            (cx1 - w1 // 2 + max(2, w1 // 8), s_y1),
            (cx2 - w2 // 2 + max(2, w2 // 8), s_y2),
            (cx2 - w2 // 2, s_y2),
        ]
        draw.polygon(sl_poly, fill=sleeve_light)

    # --- Крышка (эллипс сверху) ---
    if add_lid:
        lid_color = rng.choice(LID_COLORS)
        # Эллипс чуть шире верха стакана
        lid_w = int(top_width * 1.05)
        lid_h = max(6, int(top_width * 0.18))
        lid_x1 = top_left_x - (lid_w - top_width) // 2
        lid_y1 = top_y - lid_h // 2
        lid_x2 = lid_x1 + lid_w
        lid_y2 = lid_y1 + lid_h
        draw.ellipse([lid_x1, lid_y1, lid_x2, lid_y2], fill=lid_color,
                     outline=(0, 0, 0, 100), width=1)
        # Маленький «носик» на крышке (тёмная точка/пятно)
        sip_x = lid_x1 + lid_w // 2 + rng.randint(-3, 3)
        sip_y = lid_y1 + lid_h // 2
        draw.ellipse([sip_x - 2, sip_y - 1, sip_x + 2, sip_y + 1],
                     fill=(40, 30, 20, 200))

    # --- Пар (полупрозрачные белые линии) ---
    if add_steam:
        _draw_steam(draw, rng, center_x + skew_px, top_y - (lid_h if add_lid else 0),
                    top_width)

    # --- BBox ---
    # Включаем сам стакан + крышку, но не пар
    x_min = min(top_left_x, bot_left_x) - 1
    x_max = max(top_right_x, bot_right_x) + 1
    y_min = top_y - (lid_h if add_lid else 0)
    y_max = bottom_y
    # Обрезаем по границам изображения
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(IMG_W, x_max)
    y_max = min(IMG_H, y_max)
    return {
        "x": int(x_min),
        "y": int(y_min),
        "w": int(x_max - x_min),
        "h": int(y_max - y_min),
        "class": "cup",
    }


def _draw_steam(draw: ImageDraw.ImageDraw, rng: random.Random,
                center_x: int, top_y: int, cup_width: int) -> None:
    """Рисует 2-3 полупрозрачные волнистые линии пара над стаканом."""
    n_lines = rng.randint(2, 3)
    steam_h = rng.randint(25, 50)
    for i in range(n_lines):
        # Каждая линия — синусоида по вертикали
        x_offset = (i - n_lines // 2) * (cup_width // 4)
        amp = rng.randint(3, 7)
        phase = rng.uniform(0, 2 * np.pi)
        points = []
        for step in range(0, steam_h, 2):
            y = top_y - step - 2
            if y < 0:
                break
            x = center_x + x_offset + int(amp * np.sin(step * 0.25 + phase))
            points.append((x, y))
        if len(points) >= 2:
            # Прозрачность уменьшается кверху
            alpha = rng.randint(60, 110)
            draw.line(points, fill=(255, 255, 255, alpha),
                      width=rng.randint(2, 3), joint="curve")


def _place_cups_on_image(img: Image.Image, rng: random.Random,
                         n_cups: int) -> list[dict]:
    """Размещает n_cups стаканов на изображении, возвращает список bbox."""
    boxes: list[dict] = []
    # Запоминаем центры, чтобы стаканы не сильно перекрывались
    placed_centers: list[tuple[int, int, int]] = []  # (cx, by, top_w)

    for _ in range(n_cups):
        # Подбираем позицию с минимальным перекрытием
        for _attempt in range(15):
            cup_height = rng.randint(80, 150)
            top_width = rng.randint(60, 100)
            bottom_width = rng.randint(40, 70)
            # Нижняя кромка в нижней половине изображения (стакан «стоит» на столе)
            bottom_y = rng.randint(cup_height + 20, IMG_H - 10)
            center_x = rng.randint(top_width // 2 + 10,
                                   IMG_W - top_width // 2 - 10)
            # Проверяем расстояние до уже размещённых
            ok = True
            for (pcx, pby, ptw) in placed_centers:
                if abs(center_x - pcx) < (top_width + ptw) // 2 + 5 \
                        and abs(bottom_y - pby) < 40:
                    ok = False
                    break
            if ok:
                placed_centers.append((center_x, bottom_y, top_width))
                break

        cup_color = rng.choice(CUP_COLORS)
        skew_px = rng.randint(-6, 6)
        add_sleeve = rng.random() < 0.55  # ~55% стаканов с sleeve
        add_lid = rng.random() < 0.40     # ~40% с крышкой
        add_steam = rng.random() < 0.30   # ~30% с паром

        bbox = _draw_cup(
            img=img,
            rng=rng,
            center_x=center_x,
            bottom_y=bottom_y,
            cup_height=cup_height,
            top_width=top_width,
            bottom_width=bottom_width,
            cup_color=cup_color,
            skew_px=skew_px,
            add_sleeve=add_sleeve,
            add_lid=add_lid,
            add_steam=add_steam,
        )
        if bbox["w"] > 10 and bbox["h"] > 10:
            boxes.append(bbox)

    return boxes


# --- Основная логика ---

def generate_dataset(count: int, out_dir: Path, seed: int | None = None) -> dict:
    """Генерирует count изображений + bounding_boxes.json. Возвращает summary dict."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    rng = random.Random(seed if seed is not None else None)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Очищаем старые png в папке (кроме json)
    for old in out_dir.glob("cup_*.png"):
        try:
            old.unlink()
        except OSError:
            pass

    images_meta: list[dict] = []
    total_cups = 0

    for i in range(1, count + 1):
        # Фон
        img = _make_random_background(rng)
        # Лёгкое размытие фона (имитация глубины резкости)
        if rng.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))

        # 1–3 стакана
        n_cups = rng.choices([1, 2, 3], weights=[5, 3, 2])[0]
        boxes = _place_cups_on_image(img, rng, n_cups)
        total_cups += len(boxes)

        # Сохраняем изображение
        img_name = f"cup_{i:04d}.png"
        img_path = out_dir / img_name
        img.save(img_path, "PNG")

        images_meta.append({"image": img_name, "boxes": boxes})

        if i % 10 == 0 or i == count:
            click.echo(f"  сгенерировано {i}/{count} изображений")

    # Сохраняем ground truth JSON
    gt_path = out_dir / "bounding_boxes.json"
    gt_data = {"images": images_meta, "count": count, "total_boxes": total_cups}
    gt_path.write_text(json.dumps(gt_data, ensure_ascii=False, indent=2),
                       encoding="utf-8")

    return {
        "count": count,
        "total_cups": total_cups,
        "out_dir": str(out_dir),
        "gt_path": str(gt_path),
    }


# --- CLI ---

@click.command()
@click.option("--count", "-n", default=DEFAULT_COUNT, type=int,
              help=f"Сколько изображений сгенерировать (по умолчанию {DEFAULT_COUNT}).")
@click.option("--out", "-o", "out_dir", default=DEFAULT_OUT,
              type=click.Path(),
              help=f"Куда сохранять изображения (по умолчанию {DEFAULT_OUT}).")
@click.option("--seed", default=None, type=int,
              help="Сид для воспроизводимости (по умолчанию — случайно).")
def main(count: int, out_dir: str, seed: int | None) -> None:
    """Генератор синтетических изображений бумажных стаканов для CV-тестов."""
    click.echo(click.style(
        "COFFEE — генератор синтетических стаканов (PIL + numpy)",
        fg="cyan", bold=True,
    ))
    click.echo(f"Количество: {count}")
    click.echo(f"Выходная папка: {out_dir}")
    if seed is not None:
        click.echo(f"Сид: {seed}")
    click.echo("")

    try:
        summary = generate_dataset(count=count, out_dir=Path(out_dir), seed=seed)
    except Exception as exc:  # noqa: BLE001
        click.echo(click.style(f"Ошибка генерации: {exc}", fg="red"), err=True)
        sys.exit(1)

    click.echo("")
    click.echo(click.style("Готово!", fg="green", bold=True))
    click.echo(f"  Изображений: {summary['count']}")
    click.echo(f"  Всего стаканов: {summary['total_cups']}")
    click.echo(f"  Папка: {summary['out_dir']}")
    click.echo(f"  Ground truth: {summary['gt_path']}")


if __name__ == "__main__":
    main()
