#!/usr/bin/env python3
"""
Черновики эталонов рядом с изображениями: локальный **Tesseract** + **Docling**.

Пишет для каждого ``stem.png``:
  • ``stem.ref.draft.docling.md`` — ``export_to_markdown()`` (заголовки, таблицы, порядок чтения)
  • ``stem.ref.draft.tesseract.txt`` — плоский текст ``image_to_string`` (те же OEM/PSM/lang, что в task01)

Эти файлы **не** подхватываются бенчмарком как эталон. После правки скопируйте или переименуйте в
``stem.ref.txt`` или ``stem.ref.md`` (см. ``run_ocr_api_benchmark.load_reference``).

Зависимости:
  pip install pillow pytesseract docling

Системно: бинарь ``tesseract`` и языки (например ``rus``, ``eng``).

Пример:
  python scripts/draft_reference_from_local_ocr.py --input-dir input/data/1 --print-summary
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable


def _iter_images(input_dir: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(input_dir.glob(pattern))


def _tesseract_config(oem: int, psm: int, extra: str | None) -> str:
    parts = [f"--oem {int(oem)}", f"--psm {int(psm)}"]
    if extra:
        parts.append(extra.strip())
    return " ".join(parts)


def _run_tesseract(path: Path, *, lang: str, oem: int, psm: int, extra: str | None, scale: float) -> tuple[str, float]:
    from PIL import Image
    import pytesseract

    img = Image.open(path).convert("RGB")
    if scale != 1.0:
        w, h = img.size
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    cfg = _tesseract_config(oem, psm, extra)
    t0 = time.perf_counter()
    text = pytesseract.image_to_string(img, lang=lang, config=cfg) or ""
    elapsed = time.perf_counter() - t0
    return text, elapsed


def _prepare_scaled_png(path: Path, scale: float) -> tuple[Path, bool]:
    """Путь для Docling; второй элемент — удалить ли временный файл."""
    from PIL import Image

    if scale == 1.0:
        return path, False
    img = Image.open(path).convert("RGB")
    w, h = img.size
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    fd, name = tempfile.mkstemp(suffix=".png", prefix="draft_ref_docling_")
    import os

    os.close(fd)
    outp = Path(name)
    img.save(outp, format="PNG")
    return outp, True


def _run_docling_markdown(path: Path, *, scale: float) -> tuple[str, float]:
    from docling.document_converter import DocumentConverter

    src, is_temp = _prepare_scaled_png(path, scale)
    try:
        t0 = time.perf_counter()
        result = DocumentConverter().convert(src)
        md = ""
        try:
            md = result.document.export_to_markdown() or ""
        except Exception:
            md = ""
        elapsed = time.perf_counter() - t0
        return md, elapsed
    finally:
        if is_temp:
            try:
                src.unlink(missing_ok=True)
            except OSError:
                pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=Path("input/data/1"), help="Каталог с изображениями")
    p.add_argument("--glob", default="*.png", help='Шаблон файлов, напр. "*.png"')
    p.add_argument("--lang", default="rus+eng", help="Tesseract -l")
    p.add_argument("--oem", type=int, default=3)
    p.add_argument("--psm", type=int, default=3)
    p.add_argument(
        "--tesseract-extra-config",
        default=None,
        help="Доп. флаги tesseract, напр. '-c preserve_interword_spaces=1'",
    )
    p.add_argument("--scale", type=float, default=1.0, help="Масштаб перед обоими движками")
    p.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать существующие *.ref.draft.*",
    )
    p.add_argument(
        "--tesseract-only",
        action="store_true",
        help="Не вызывать Docling (только черновик Tesseract)",
    )
    p.add_argument(
        "--docling-only",
        action="store_true",
        help="Не вызывать Tesseract (только черновик Docling)",
    )
    p.add_argument(
        "--combined-md",
        action="store_true",
        help="Дополнительно писать stem.ref.draft.combined.md (Docling + Tesseract в одном файле)",
    )
    p.add_argument("--print-summary", action="store_true", help="Кратко по файлам в stdout")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.tesseract_only and args.docling_only:
        raise SystemExit("Нельзя одновременно --tesseract-only и --docling-only")

    input_dir = args.input_dir
    if not input_dir.is_dir():
        raise SystemExit(f"Нет каталога: {input_dir.resolve()}")

    if not args.docling_only:
        try:
            import pytesseract  # noqa: F401
        except ImportError as e:
            raise SystemExit("Нужен pytesseract: pip install pytesseract pillow\n" + repr(e)) from e

    if not args.tesseract_only:
        try:
            import docling  # noqa: F401
        except ImportError as e:
            raise SystemExit("Нужен docling: pip install docling pillow\n" + repr(e)) from e

    n_ok = 0
    n_skip = 0
    errors: list[str] = []

    for img in _iter_images(input_dir, args.glob):
        if not img.is_file():
            continue
        stem = img.stem
        p_dl = img.parent / f"{stem}.ref.draft.docling.md"
        p_ts = img.parent / f"{stem}.ref.draft.tesseract.txt"
        p_cb = img.parent / f"{stem}.ref.draft.combined.md"

        need_dl = not args.tesseract_only
        need_ts = not args.docling_only

        if not args.force:
            if need_dl and need_ts and p_dl.is_file() and p_ts.is_file():
                if not args.combined_md or p_cb.is_file():
                    n_skip += 1
                    continue
            elif need_dl and not need_ts and p_dl.is_file():
                n_skip += 1
                continue
            elif need_ts and not need_dl and p_ts.is_file():
                n_skip += 1
                continue

        md_text = ""
        ts_text = ""
        t_dl = 0.0
        t_ts = 0.0

        if need_dl:
            try:
                md_text, t_dl = _run_docling_markdown(img, scale=args.scale)
            except Exception as e:
                errors.append(f"{img.name}: docling: {e!r}")
                md_text = f"[Docling error: {e!r}]\n"

        if need_ts:
            try:
                ts_text, t_ts = _run_tesseract(
                    img,
                    lang=args.lang,
                    oem=args.oem,
                    psm=args.psm,
                    extra=args.tesseract_extra_config,
                    scale=args.scale,
                )
            except Exception as e:
                errors.append(f"{img.name}: tesseract: {e!r}")
                ts_text = f"[Tesseract error: {e!r}]\n"

        if need_dl:
            p_dl.write_text(md_text.rstrip() + ("\n" if md_text.strip() else ""), encoding="utf-8")
        if need_ts:
            p_ts.write_text(ts_text.rstrip() + ("\n" if ts_text.strip() else ""), encoding="utf-8")

        if args.combined_md:
            parts = [
                "# Docling (структура, markdown)\n",
                "\n",
                md_text.rstrip() + "\n\n",
                "---\n\n",
                "# Tesseract (плоский текст)\n",
                "\n",
                ts_text.rstrip() + "\n",
            ]
            p_cb.write_text("".join(parts), encoding="utf-8")

        n_ok += 1
        if args.print_summary:
            bits = [img.name]
            if need_dl:
                bits.append(f"docling {t_dl:.2f}s/{len(md_text)}ch")
            if need_ts:
                bits.append(f"tesseract {t_ts:.2f}s/{len(ts_text)}ch")
            print(" | ".join(bits))

    if n_ok:
        print(
            f"Готово: записано черновиков для {n_ok} файлов; "
            f"пропущено (уже есть, без --force): {n_skip}. "
            "Дальше: отредактировать и сохранить как <stem>.ref.txt или <stem>.ref.md рядом с PNG."
        )
    else:
        print(
            f"Файлы не записаны (пропущено {n_skip}). "
            "Укажите --force или удалите *.ref.draft.* рядом с изображениями."
        )

    for err in errors:
        print("Предупреждение:", err, file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
