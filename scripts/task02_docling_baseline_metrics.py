#!/usr/bin/env python3
"""
Задача 02: прогон Docling по изображениям (и при необходимости PDF) — те же метрики, что в task01.

Зависимости:
  pip install docling pillow

Вход: те же каталоги/шаблоны, что для Tesseract (по умолчанию PNG в input/data/1).

Поля JSON совпадают с task01 (`OcrMetrics`), чтобы сравнивать JSONL построчно или склеивать внешним скриптом.

Семантика полей, отличающаяся от Tesseract:
  • lang — фиксировано «docling» (это не язык OCR).
  • oem / psm — -1 (не применимо).
  • tesseract_config — строка с пометкой пайплайна Docling (имя поля сохранено для совместимости схемы).
  • mean_word_conf / median / frac_* — из **Docling ConfidenceReport** (качество страницы/документа,
    агрегат layout/OCR/table/parse, 0–1), переведено в 0–100 **только для визуального сравнения шкалы**
    с Tesseract; это не «уверенность по словам».
  • block_count — число элементов, возвращаемых DoclingDocument.iterate_items() (структурные узлы,
    не эквивалент блокам Tesseract, но сравнимо между прогонами Docling).

Запуск (пример):
  python scripts/task02_docling_baseline_metrics.py \\
    --input-dir "input/data/1" \\
    --output-jsonl output/task02_docling_metrics.jsonl \\
    --print-summary-ru
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Конфиг по умолчанию
# ---------------------------------------------------------------------------

DEFAULT_INPUT_DIR = Path("input/data/1")
DEFAULT_GLOB = "*.png"
ENGINE_LABEL = "docling"
PLACEHOLDER_OEM = -1
PLACEHOLDER_PSM = -1
PIPELINE_NOTE_DEFAULT = "DocumentConverter()"


# ---------------------------------------------------------------------------
# Та же схема метрик, что в task01_ocr_tesseract_baseline_metrics.py
# ---------------------------------------------------------------------------


@dataclass
class OcrMetrics:
    """Сводка по одному файлу (схема идентична task01)."""

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
    lines_with_tab_ge2: int
    lines_with_pipe_ge2: int
    pipe_char_count: int
    tableish_line_ratio: float
    chars_per_megapixel: float


# ---------------------------------------------------------------------------
# Утилиты текста / таблиц (как в task01)
# ---------------------------------------------------------------------------


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


def _finite_float(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _prepare_source_image(path: Path, scale: float) -> tuple[Path, bool, int, int]:
    """
    Возвращает (путь для convert, нужно_ли_удалить, width, height).
    При scale != 1 пишет временный PNG.
    """
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if scale != 1.0:
        w, h = img.size
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    w, h = img.size
    if scale == 1.0:
        return path, False, w, h
    fd, name = tempfile.mkstemp(suffix=".png", prefix="docling_scale_")
    try:
        import os

        os.close(fd)
        outp = Path(name)
        img.save(outp, format="PNG")
        return outp, True, w, h
    except Exception:
        import os

        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _confidence_to_wordlike_fields(confidence) -> tuple[float | None, float | None, float | None, float | None]:
    """
    Маппинг Docling ConfidenceReport → поля mean/median/frac (шкала 0–100).
    Использует per-page mean_score и документный mean_score.
    """
    mean_doc = _finite_float(getattr(confidence, "mean_score", None))
    mean_word = round(mean_doc * 100.0, 2) if mean_doc is not None else None

    page_scores: list[float] = []
    pages = getattr(confidence, "pages", None) or {}
    for p in pages.values():
        ms = _finite_float(getattr(p, "mean_score", None))
        if ms is not None:
            page_scores.append(ms * 100.0)

    med = round(statistics.median(page_scores), 2) if page_scores else None

    low60 = sum(1 for s in page_scores if s < 60) / len(page_scores) if page_scores else None
    low30 = sum(1 for s in page_scores if s < 30) / len(page_scores) if page_scores else None

    if low60 is None and mean_word is not None:
        low60 = 1.0 if mean_word < 60 else 0.0
    if low30 is None and mean_word is not None:
        low30 = 1.0 if mean_word < 30 else 0.0

    return mean_word, med, round(low60, 4) if low60 is not None else None, round(low30, 4) if low30 is not None else None


def _docling_iterate_item_count(document) -> int:
    try:
        return sum(1 for _ in document.iterate_items())
    except Exception:
        return 0


def _run_docling_convert(source_path: Path, pipeline_note: str):
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    t0 = time.perf_counter()
    result = converter.convert(source_path)
    elapsed = time.perf_counter() - t0
    return result, elapsed, pipeline_note


def collect_metrics_for_file(
    path: Path,
    *,
    scale: float,
    pipeline_note: str,
) -> OcrMetrics:
    src, is_temp, w, h = _prepare_source_image(path, scale)
    try:
        result, elapsed, note = _run_docling_convert(src, pipeline_note)
        doc = result.document
        text = ""
        try:
            text = doc.export_to_markdown() or ""
        except Exception:
            text = ""

        lines = text.splitlines()
        char_count = len(text)
        line_count = len(lines)
        word_count = _count_words_from_text(text)
        tab_ln, pipe_ln, pipe_ch, tabish_ratio = _tableish_stats(text)
        mpx = (w * h) / 1_000_000.0
        cpm = char_count / mpx if mpx > 0 else 0.0

        mean_w, med_w, f60, f30 = _confidence_to_wordlike_fields(result.confidence)

        return OcrMetrics(
            file=str(path),
            image_width=w,
            image_height=h,
            elapsed_sec=round(elapsed, 4),
            lang=ENGINE_LABEL,
            oem=PLACEHOLDER_OEM,
            psm=PLACEHOLDER_PSM,
            tesseract_config=f"docling:{note}",
            char_count=char_count,
            line_count=line_count,
            word_count=word_count,
            block_count=_docling_iterate_item_count(doc),
            mean_word_conf=mean_w,
            median_word_conf=med_w,
            frac_word_conf_below_60=f60,
            frac_word_conf_below_30=f30,
            empty_text=(char_count == 0 or not text.strip()),
            lines_with_tab_ge2=tab_ln,
            lines_with_pipe_ge2=pipe_ln,
            pipe_char_count=pipe_ch,
            tableish_line_ratio=round(tabish_ratio, 4),
            chars_per_megapixel=round(cpm, 2),
        )
    finally:
        if is_temp:
            try:
                src.unlink(missing_ok=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Обход файлов и запись результатов
# ---------------------------------------------------------------------------


def iter_inputs(input_dir: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(input_dir.glob(pattern))


def write_jsonl(path: Path, rows: list[OcrMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def summarize(rows: list[OcrMetrics]) -> dict:
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
    lines: list[str] = []
    lines.append("=== Сводка прогона Docling ===")
    lines.append("")
    lines.append("Общие показатели:")
    lines.append(
        f"  • Обработано файлов: {agg.get('n_files', 0)} шт. "
        f"(полностью пустой извлечённый текст: {agg.get('empty_text_files', 0)})."
    )
    lines.append(
        f"  • Время конвертации: всего {agg.get('total_elapsed_sec')} с, "
        f"в среднем {agg.get('mean_elapsed_sec_per_file')} с на файл."
    )
    lines.append(
        f"  • Символов в export_to_markdown(): всего {agg.get('total_chars')}, "
        f"в среднем {agg.get('mean_chars_per_file')} на файл."
    )
    mc = agg.get("mean_word_conf_over_files")
    if mc is not None:
        lines.append(
            f"  • Поле mean_word_conf (0–100): усреднённое по файлам **{mc}** — "
            f"это Docling ConfidenceReport.mean_score × 100 (качество пайплайна на странице/документе), "
            f"а не построчная уверенность Tesseract по словам."
        )
    else:
        lines.append("  • Поле mean_word_conf: нет конечного confidence (NaN или отсутствует).")
    lines.append("")
    if rows:
        lines.append("По файлам (имя | время с | символов | слов | conf×100 | элементов iterate_items):")
        for r in rows:
            name = Path(r.file).name
            conf = f"{r.mean_word_conf:.1f}" if r.mean_word_conf is not None else "—"
            lines.append(
                f"  • {name} | {r.elapsed_sec} с | {r.char_count} симв. | "
                f"{r.word_count} слов | score≈{conf} | узлов: {r.block_count}"
            )
    lines.append("")
    lines.append("Подсказка: построчные записи — в JSONL (--output-jsonl); сравнивайте с task01 на одном наборе файлов.")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Каталог с файлами")
    p.add_argument("--glob", default=DEFAULT_GLOB, help='Шаблон, напр. "*.png" или "*.pdf"')
    p.add_argument(
        "--pipeline-note",
        default=PIPELINE_NOTE_DEFAULT,
        help="Короткая метка пайплайна (пишется в tesseract_config после docling:)",
    )
    p.add_argument("--scale", type=float, default=1.0, help="Масштаб изображения перед конвертацией (>1 — увеличить)")
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("output/task02_docling_metrics.jsonl"),
        help="Куда писать метрики по файлам (JSON Lines, схема как task01)",
    )
    p.add_argument("--print-summary", action="store_true", help="Сводка в stdout (JSON)")
    p.add_argument("--print-summary-ru", action="store_true", help="Сводка по-русски")
    return p.parse_args()


def main() -> int:
    try:
        import docling  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "Не установлен пакет docling. Установите: pip install docling pillow\n" f"({e})"
        ) from e

    args = parse_args()
    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        raise SystemExit(f"Нет каталога: {input_dir.resolve()}")

    rows: list[OcrMetrics] = []
    for p in iter_inputs(input_dir, args.glob):
        if not p.is_file():
            continue
        rows.append(
            collect_metrics_for_file(
                p,
                scale=args.scale,
                pipeline_note=args.pipeline_note,
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
