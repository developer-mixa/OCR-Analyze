#!/usr/bin/env python3
"""
Задача 03: сравнение текста с внешнего OCR (или любого источника) с эталонным текстом.

Зависимости:
  pip install jiwer

Считает CER (Character Error Rate) и производные величины для вашей таблицы метрик.
Поля вроде Unit Test Rate / Final Score здесь не «запускаются» — только то, что выводится из
пары строк; остальное заполняйте вручную или другим пайплайном.

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
import re
import sys
import unicodedata
from pathlib import Path


def _read_text(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"Файл не найден: {p.resolve()}")
    return p.read_text(encoding="utf-8")


def _normalize(s: str, mode: str) -> str:
    if mode == "none":
        return s
    t = s
    if mode in ("nfkc", "nfkc_ws", "nfkc_ws_lower"):
        t = unicodedata.normalize("NFKC", t)
    if mode in ("nfkc_ws", "nfkc_ws_lower", "ws", "ws_lower"):
        t = re.sub(r"\s+", " ", t).strip()
    if mode in ("nfkc_ws_lower", "ws_lower"):
        t = t.lower()
    return t


def _cer_jiwer(ref: str, hyp: str) -> float:
    import jiwer

    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0  # любой ненулевой hyp при пустом эталоне — 100% ошибка по символам

    # jiwer: первый аргумент — референс, второй — гипотеза
    return float(jiwer.cer(ref, hyp))


def _wer_jiwer(ref: str, hyp: str) -> float:
    import jiwer

    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0

    return float(jiwer.wer(ref, hyp))


def build_report(
    *,
    ref_raw: str,
    hyp_raw: str,
    normalize: str,
    model: str,
    internal_parser: str | None,
    extra_comment: str | None,
) -> dict:
    ref = _normalize(ref_raw, normalize)
    hyp = _normalize(hyp_raw, normalize)

    cer = _cer_jiwer(ref, hyp)
    wer = _wer_jiwer(ref, hyp)
    char_accuracy = max(0.0, min(1.0, 1.0 - cer))
    word_accuracy = max(0.0, min(1.0, 1.0 - wer))

    # Грубая оценка токенов по гипотезе (без API эмбеддера): слова + символы для справки
    hyp_words = len(hyp.split())
    hyp_chars = len(hyp)

    # Final Score: явная договорённость по умолчанию — 100 * (1 - CER), округлить
    final_score = round(100.0 * (1.0 - cer), 2)

    row = {
        "Model": model,
        "Final Score": final_score,
        "Accuracy": round(char_accuracy, 6),
        "CER": round(cer, 6),
        "Unit Test Rate": None,
        "Внутренний парсер": internal_parser or "",
        "Токены": {
            "hypothesis_chars": hyp_chars,
            "hypothesis_words_approx": hyp_words,
            "note": "грубые счётчики по нормализованной гипотезе; для биллинга модели используйте tokenizer целевой модели",
        },
        "Доп. Комментарии": extra_comment or "",
        "Скорость": None,
        "_diagnostics": {
            "normalize": normalize,
            "reference_chars": len(ref),
            "hypothesis_chars": len(hyp),
            "WER": round(wer, 6),
            "word_accuracy_1_minus_wer": round(word_accuracy, 6),
        },
    }
    return row


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
