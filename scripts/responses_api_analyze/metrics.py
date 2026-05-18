"""Метрики как в scripts/model_analyze/task03_compare_ocr_text_to_reference.py (без импорта по пути)."""
from __future__ import annotations

import re
import unicodedata


def normalize_text(s: str, mode: str) -> str:
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


def char_error_rate(ref: str, hyp: str) -> float:
    import jiwer

    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    return float(jiwer.cer(ref, hyp))


def build_metric_row(
    *,
    ref_raw: str,
    hyp_raw: str,
    normalize: str,
    model: str,
    internal_parser: str | None,
    extra_comment: str | None,
) -> dict:
    import jiwer

    ref = normalize_text(ref_raw, normalize)
    hyp = normalize_text(hyp_raw, normalize)
    if not ref and not hyp:
        cer = 0.0
        wer = 0.0
    elif not ref:
        cer = 1.0
        wer = 1.0
    else:
        cer = float(jiwer.cer(ref, hyp))
        wer = float(jiwer.wer(ref, hyp))
    char_accuracy = max(0.0, min(1.0, 1.0 - cer))
    hyp_words = len(hyp.split())
    hyp_chars = len(hyp)
    final_score = round(100.0 * (1.0 - cer), 2)
    return {
        "Model": model,
        "Final Score": final_score,
        "Accuracy": round(char_accuracy, 6),
        "CER": round(cer, 6),
        "Unit Test Rate": None,
        "Внутренний парсер": internal_parser or "",
        "Токены": {
            "hypothesis_chars": hyp_chars,
            "hypothesis_words_approx": hyp_words,
            "note": "грубые счётчики по нормализованной гипотезе; токены биллинга — из usage API если есть",
        },
        "Доп. Комментарии": extra_comment or "",
        "Скорость": None,
        "_diagnostics": {
            "normalize": normalize,
            "reference_chars": len(ref),
            "hypothesis_chars": len(hyp),
            "WER": round(wer, 6),
        },
    }
