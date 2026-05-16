#!/usr/bin/env python3
"""
Задача 01: базовый прогон Tesseract по набору изображений и сбор сравнимых метрик.

Зависимости (pip):
  pip install pillow pytesseract

Системно должен быть установлен бинарь `tesseract` и языковые пакеты (например rus, eng).

Запуск (пример):
  python scripts/task01_ocr_tesseract_baseline_metrics.py \\
    --input-dir "input/data/1" \\
    --output-jsonl output/task01_tesseract_metrics.jsonl \\
    --print-summary-ru

Скрипт сам ничего не «тестирует» без вашего запуска — только код пайплайна и метрик.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Конфиг по умолчанию (можно переопределить флагами CLI)
# ---------------------------------------------------------------------------

DEFAULT_INPUT_DIR = Path("input/data/1")
DEFAULT_GLOB = "*.png"
DEFAULT_LANG = "rus+eng"
DEFAULT_OEM = 3  # LSTM по умолчанию в современных сборках
DEFAULT_PSM = 3  # полностью автоматическая сегментация страницы


# ---------------------------------------------------------------------------
# Модели данных
# ---------------------------------------------------------------------------


@dataclass
class OcrMetrics:
    """Сводка по одному файлу после OCR (без эталона — инженерные и прокси-метрики)."""

    file: str
    image_width: int
    image_height: int
    elapsed_sec: float
    lang: str
    oem: int
    psm: int
    tesseract_config: str
    char_count: int
    line_count: int
    word_count: int
    block_count: int
    mean_word_conf: float | None
    median_word_conf: float | None
    frac_word_conf_below_60: float | None
    frac_word_conf_below_30: float | None
    empty_text: bool
    # прокси под «табличность» по сырому тексту
    lines_with_tab_ge2: int
    lines_with_pipe_ge2: int
    pipe_char_count: int
    tableish_line_ratio: float
    # грубый прокси «много текста / мало текста на пиксель»
    chars_per_megapixel: float


# ---------------------------------------------------------------------------
# Изображение и (опционально) препроцессинг
# ---------------------------------------------------------------------------


def load_image(path: Path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def maybe_preprocess(img, scale: float):
    """Простой масштаб; при scale=1.0 возвращает исходное изображение."""
    from PIL import Image

    if scale == 1.0:
        return img
    w, h = img.size
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# Tesseract
# ---------------------------------------------------------------------------


def _build_tesseract_config(oem: int, psm: int, extra: str | None) -> str:
    parts = [f"--oem {int(oem)}", f"--psm {int(psm)}"]
    if extra:
        parts.append(extra.strip())
    return " ".join(parts)


def run_tesseract_on_image(img, lang: str, oem: int, psm: int, extra_config: str | None):
    """
    Возвращает (plain_text, tsv_dict) через pytesseract.
    tsv_dict — словари списков как в Output.DICT.
    """
    import pytesseract
    from pytesseract import Output

    cfg = _build_tesseract_config(oem, psm, extra_config)
    text = pytesseract.image_to_string(img, lang=lang, config=cfg)
    data = pytesseract.image_to_data(img, lang=lang, config=cfg, output_type=Output.DICT)
    return text, data


def _word_level_confs(data: dict) -> list[float]:
    """Извлекает уверенности по словам (строки level==5 в TSV; pytesseract кладёт level в data['level'])."""
    levels = data.get("level") or []
    confs = data.get("conf") or []
    texts = data.get("text") or []
    out: list[float] = []
    for lvl, c, t in zip(levels, confs, texts):
        if int(lvl) != 5:
            continue
        if not (t or "").strip():
            continue
        try:
            ci = int(c)
        except (TypeError, ValueError):
            continue
        if ci < 0:  # -1 = нет оценки
            continue
        out.append(float(ci))
    return out


def _block_count_from_data(data: dict) -> int:
    """Число уникальных block_num на словах (level 5)."""
    levels = data.get("level") or []
    blocks = data.get("block_num") or []
    return len({int(b) for lvl, b in zip(levels, blocks) if int(lvl) == 5})


def _count_words_from_text(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def _tableish_stats(text: str) -> tuple[int, int, int, float]:
    lines = text.splitlines()
    if not lines:
        return 0, 0, 0, 0.0
    tab_lines = sum(1 for ln in lines if ln.count("\t") >= 2)
    pipe_lines = sum(1 for ln in lines if ln.count("|") >= 2)
    pipes = text.count("|")
    ratio = (tab_lines + pipe_lines) / max(1, len(lines))
    return tab_lines, pipe_lines, pipes, ratio


def collect_metrics_for_file(
    path: Path,
    *,
    lang: str,
    oem: int,
    psm: int,
    extra_config: str | None,
    scale: float,
) -> OcrMetrics:
    img0 = load_image(path)
    img = maybe_preprocess(img0, scale)
    w, h = img.size
    cfg = _build_tesseract_config(oem, psm, extra_config)

    t0 = time.perf_counter()
    text, data = run_tesseract_on_image(img, lang=lang, oem=oem, psm=psm, extra_config=extra_config)
    elapsed = time.perf_counter() - t0

    lines = text.splitlines()
    char_count = len(text)
    line_count = len(lines)
    word_count = _count_words_from_text(text)
    confs = _word_level_confs(data)

    mean_c = statistics.mean(confs) if confs else None
    med_c = statistics.median(confs) if confs else None
    low60 = sum(1 for c in confs if c < 60) / len(confs) if confs else None
    low30 = sum(1 for c in confs if c < 30) / len(confs) if confs else None

    tab_ln, pipe_ln, pipe_ch, tabish_ratio = _tableish_stats(text)
    mpx = (w * h) / 1_000_000.0
    cpm = char_count / mpx if mpx > 0 else 0.0

    return OcrMetrics(
        file=str(path),
        image_width=w,
        image_height=h,
        elapsed_sec=round(elapsed, 4),
        lang=lang,
        oem=oem,
        psm=psm,
        tesseract_config=cfg,
        char_count=char_count,
        line_count=line_count,
        word_count=word_count,
        block_count=_block_count_from_data(data),
        mean_word_conf=round(mean_c, 2) if mean_c is not None else None,
        median_word_conf=round(med_c, 2) if med_c is not None else None,
        frac_word_conf_below_60=round(low60, 4) if low60 is not None else None,
        frac_word_conf_below_30=round(low30, 4) if low30 is not None else None,
        empty_text=(char_count == 0 or not text.strip()),
        lines_with_tab_ge2=tab_ln,
        lines_with_pipe_ge2=pipe_ln,
        pipe_char_count=pipe_ch,
        tableish_line_ratio=round(tabish_ratio, 4),
        chars_per_megapixel=round(cpm, 2),
    )


# ---------------------------------------------------------------------------
# Обход файлов и запись результатов
# ---------------------------------------------------------------------------


def iter_images(input_dir: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(input_dir.glob(pattern))


def write_jsonl(path: Path, rows: list[OcrMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def summarize(rows: list[OcrMetrics]) -> dict:
    """Агрегаты по всему прогону (удобно смотреть глазами)."""
    if not rows:
        return {"n_files": 0}
    times = [r.elapsed_sec for r in rows]
    chars = [r.char_count for r in rows]
    empties = sum(1 for r in rows if r.empty_text)
    confs = [r.mean_word_conf for r in rows if r.mean_word_conf is not None]
    return {
        "n_files": len(rows),
        "total_elapsed_sec": round(sum(times), 3),
        "mean_elapsed_sec_per_file": round(statistics.mean(times), 4),
        "total_chars": sum(chars),
        "mean_chars_per_file": round(statistics.mean(chars), 1),
        "empty_text_files": empties,
        "mean_word_conf_over_files": round(statistics.mean(confs), 2) if confs else None,
    }


def format_summary_ru(rows: list[OcrMetrics], agg: dict) -> str:
    """Человекочитаемая сводка на русском (агрегаты + кратко по каждому файлу)."""
    lines: list[str] = []
    lines.append("=== Сводка прогона Tesseract ===")
    lines.append("")
    lines.append("Общие показатели:")
    lines.append(
        f"  • Обработано изображений: {agg.get('n_files', 0)} шт. "
        f"(пустой текст ни у одного файла: «пустых» = {agg.get('empty_text_files', 0)})."
    )
    lines.append(
        f"  • Время OCR: всего {agg.get('total_elapsed_sec')} с, "
        f"в среднем {agg.get('mean_elapsed_sec_per_file')} с на одно изображение."
    )
    lines.append(
        f"  • Распознано символов: всего {agg.get('total_chars')}, "
        f"в среднем {agg.get('mean_chars_per_file')} символов на изображение."
    )
    mc = agg.get("mean_word_conf_over_files")
    if mc is not None:
        lines.append(
            f"  • Средняя уверенность по словам (Tesseract conf, 0–100): {mc} "
            f"— это среднее арифметическое **пофайловых** средних conf по словам "
            f"(не взвешенное по числу слов во всём наборе)."
        )
        lines.append(
            "    Ориентир: выше ~80 обычно «довольно уверенно» на чистых скриншотах; "
            "на сканах и мелком шрифте типично ниже. Сравнивайте движки между собой на одних и тех же файлах."
        )
    else:
        lines.append("  • Средняя уверенность по словам: нет данных (нет слов с оценкой conf).")
    lines.append("")
    if rows:
        lines.append("По файлам (имя | время с | символов | слов | сред. conf слов | блоков):")
        for r in rows:
            name = Path(r.file).name
            conf = f"{r.mean_word_conf:.1f}" if r.mean_word_conf is not None else "—"
            lines.append(
                f"  • {name} | {r.elapsed_sec} с | {r.char_count} симв. | "
                f"{r.word_count} слов | conf≈{conf} | блоков: {r.block_count}"
            )
    lines.append("")
    lines.append("Подсказка: подробные поля по каждому файлу — в JSONL (--output-jsonl).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Каталог с изображениями")
    p.add_argument("--glob", default=DEFAULT_GLOB, help='Шаблон файлов, напр. "*.png" или "*.{png,jpg}"')
    p.add_argument("--lang", default=DEFAULT_LANG)
    p.add_argument("--oem", type=int, default=DEFAULT_OEM)
    p.add_argument("--psm", type=int, default=DEFAULT_PSM)
    p.add_argument(
        "--tesseract-extra-config",
        default=None,
        help="Доп. флаги tesseract, напр. '-c preserve_interword_spaces=1'",
    )
    p.add_argument("--scale", type=float, default=1.0, help="Масштаб изображения перед OCR (>1 — увеличить)")
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("output/task01_tesseract_metrics.jsonl"),
        help="Куда писать метрики по файлам (JSON Lines)",
    )
    p.add_argument(
        "--print-summary",
        action="store_true",
        help="Печатать краткую сводку в stdout (JSON)",
    )
    p.add_argument(
        "--print-summary-ru",
        action="store_true",
        help="Печатать ту же сводку человекочитаемо по-русски (плюс строки по каждому файлу)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        raise SystemExit(f"Нет каталога: {input_dir.resolve()}")

    rows: list[OcrMetrics] = []
    for img_path in iter_images(input_dir, args.glob):
        if not img_path.is_file():
            continue
        rows.append(
            collect_metrics_for_file(
                img_path,
                lang=args.lang,
                oem=args.oem,
                psm=args.psm,
                extra_config=args.tesseract_extra_config,
                scale=args.scale,
            )
        )

    write_jsonl(args.output_jsonl, rows)
    agg = summarize(rows)
    if args.print_summary:
        print(json.dumps(agg, ensure_ascii=False, indent=2))
    if args.print_summary_ru:
        print(format_summary_ru(rows, agg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
