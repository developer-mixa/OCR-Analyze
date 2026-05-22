#!/usr/bin/env python3
"""
Задача 03: сравнение текста с внешнего OCR (или любого источника) с эталонным текстом.

Зависимости:
  pip install jiwer

Считает CER/WER (jiwer), **Final Score** как сумму частей (см. ``scripts/responses_api_analyze/metrics.py``),
плюс поля таблицы метрик. ``Unit Test Rate`` не заполняется.

Примеры:
  python scripts/task03_compare_ocr_text_to_reference.py \\
    --reference ref.txt --hypothesis ocr_out.txt --model "VendorOCR_v1"

  echo "эталон" > ref.txt && echo "эталон!" > hyp.txt
  python scripts/task03_compare_ocr_text_to_reference.py -r ref.txt -H hyp.txt --model test

  cat ocr.txt | python scripts/task03_compare_ocr_text_to_reference.py -r ref.txt --hypothesis -
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _read_text(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"Файл не найден: {p.resolve()}")
    return p.read_text(encoding="utf-8")


def build_report(
    *,
    ref_raw: str,
    hyp_raw: str,
    normalize: str,
    model: str,
    internal_parser: str | None,
    extra_comment: str | None,
) -> dict:
    mp = Path(__file__).resolve().parents[1] / "responses_api_analyze"
    if str(mp) not in sys.path:
        sys.path.insert(0, str(mp))
    from metrics import build_metric_row

    return build_metric_row(
        ref_raw=ref_raw,
        hyp_raw=hyp_raw,
        normalize=normalize,
        model=model,
        internal_parser=internal_parser,
        extra_comment=extra_comment,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-r", "--reference", required=True, help="Файл эталонного текста (UTF-8) или - для stdin")
    p.add_argument(
        "-H",
        "--hypothesis",
        required=True,
        help="Файл текста с OCR/сервиса (UTF-8) или - для stdin",
    )
    p.add_argument("--model", default="unknown", help="Имя модели/сервиса для колонки Model")
    p.add_argument("--internal-parser", default=None, help="Подпись внутреннего парсера, если есть")
    p.add_argument("--comment", default=None, help="Текст в «Доп. Комментарии»")
    p.add_argument(
        "--normalize",
        choices=("none", "nfkc", "nfkc_ws", "nfkc_ws_lower", "ws", "ws_lower"),
        default="nfkc_ws",
        help="Нормализация обеих строк перед CER/WER: NFKC, схлопывание пробелов, опционально lower",
    )
    p.add_argument("--pretty", action="store_true", help="Человекочитаемый JSON с отступами")
    return p.parse_args()


def main() -> int:
    try:
        import jiwer  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "Нужен пакет jiwer: pip install jiwer\n" f"({e})"
        ) from e

    args = parse_args()
    if args.reference == "-" and args.hypothesis == "-":
        raise SystemExit("Нельзя читать оба текста из stdin одновременно.")

    ref_raw = _read_text(args.reference)
    hyp_raw = _read_text(args.hypothesis)

    report = build_report(
        ref_raw=ref_raw,
        hyp_raw=hyp_raw,
        normalize=args.normalize,
        model=args.model,
        internal_parser=args.internal_parser,
        extra_comment=args.comment,
    )

    indent = 2 if args.pretty else None
    print(json.dumps(report, ensure_ascii=False, indent=indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
