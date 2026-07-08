#!/usr/bin/env python3
"""monitor.py — thin-обёртка над common.monitor для удобства.

Запуск из integration/::

    python integration/monitor.py
    python integration/monitor.py --topics voice.command,cv.detection

Функциональность идентична ``common/monitor.py`` — это просто алиас,
чтобы не было путаницы при запуске из разных мест.
"""

from __future__ import annotations

# --- Импорты ---
import sys
from pathlib import Path

# Локальные импорты — добавляем корень проекта в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.monitor import main  # noqa: E402

if __name__ == "__main__":
    main()
