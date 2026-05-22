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


def composite_final_score(
    cer: float,
    wer: float,
    *,
    ref_chars: int,
    hyp_chars: int,
    ref_words: int,
    hyp_words: int,
) -> tuple[float, dict[str, float | str]]:
    """
    Итоговый балл 0..100 как **сумма** частей (без учёта времени).

    **40**×(1−min(CER,1)) + **35**×(1−min(WER,1))
    + **12.5**×(1−min(rel_err_chars,1)) + **12.5**×(1−min(rel_err_words,1)),
    где rel_err — относительное расхождение длин нормализованного ref и hyp.
    """
    den_c = max(ref_chars, 1)
    rel_c = abs(ref_chars - hyp_chars) / float(den_c)
    den_w = max(ref_words, 1)
    rel_w = abs(ref_words - hyp_words) / float(den_w)

    w_cer, w_wer, w_char, w_word = 40.0, 35.0, 12.5, 12.5

    s_cer = w_cer * max(0.0, 1.0 - min(cer, 1.0))
    s_wer = w_wer * max(0.0, 1.0 - min(wer, 1.0))
    s_char = w_char * max(0.0, 1.0 - min(rel_c, 1.0))
    s_word = w_word * max(0.0, 1.0 - min(rel_w, 1.0))
    total = round(s_cer + s_wer + s_char + s_word, 2)
    parts: dict[str, float | str] = {
        "from_cer": round(s_cer, 4),
        "from_wer": round(s_wer, 4),
        "from_char_len_ratio": round(s_char, 4),
        "from_word_len_ratio": round(s_word, 4),
        "weights": "cer=40, wer=35, Δchars=12.5, Δwords=12.5; время в балл не входит",
    }
    return total, parts


def build_metric_row(
    *,
    ref_raw: str,
    hyp_raw: str,
    normalize: str,
    model: str,
    internal_parser: str | None,
    extra_comment: str | None,
    mean_elapsed_sec: float | None = None,
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
    ref_words = len(ref.split())
    hyp_words = len(hyp.split())
    hyp_chars = len(hyp)
    ref_chars = len(ref)
    word_accuracy = max(0.0, min(1.0, 1.0 - wer))

    final_score, score_parts = composite_final_score(
        cer,
        wer,
        ref_chars=ref_chars,
        hyp_chars=hyp_chars,
        ref_words=ref_words,
        hyp_words=hyp_words,
    )
    diag: dict = {
        "normalize": normalize,
        "reference_chars": ref_chars,
        "hypothesis_chars": hyp_chars,
        "reference_words_approx": ref_words,
        "WER": round(wer, 6),
        "word_accuracy_1_minus_wer_clamped": round(max(0.0, min(1.0, 1.0 - min(wer, 1.0))), 6),
        "word_accuracy_1_minus_wer": round(word_accuracy, 6),
        "final_score_parts": score_parts,
    }
    if mean_elapsed_sec is not None:
        diag["benchmark_mean_elapsed_sec"] = round(float(mean_elapsed_sec), 4)
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
        "_diagnostics": diag,
    }
