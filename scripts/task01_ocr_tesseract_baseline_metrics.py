#!/usr/bin/env python3
"""Точка входа сохранена для совместимости — реализация в model_analyze/."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "model_analyze" / "task01_ocr_tesseract_baseline_metrics.py"
    raise SystemExit(subprocess.call([sys.executable, str(script), *sys.argv[1:]]))
