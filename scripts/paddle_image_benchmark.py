#!/usr/bin/env python3
"""
Прогон **PaddleOCR** по изображениям (PNG/JPEG/…) и те же метрики против эталонов, что у ``mineru_image_benchmark``.

Пишет плоский текст в ``hypotheses/paddle/<stem>.txt``, JSONL ``paddle_runs.jsonl``,
сводку ``paddle_summaries.json`` (поля как в ``metrics.build_metric_row``),
плюс ``paddle_hypotheses_raw.json`` / ``paddle_hypotheses_concat.txt`` и ``paddleocr._outputs``.

**Эталоны:** ``<stem>.ref.txt`` → ``<stem>.ref.md`` → ``<stem>.txt`` → ``<stem>.md``.

**Зависимости:** ``pip install jiwer paddlepaddle paddleocr`` (GPU: см. https://www.paddlepaddle.org.cn/install/quick )

В **PaddleOCR 3.x** код языка ``cyrillic`` как у MinerU **не поддерживается**; для русского текста используйте ``--lang ru`` и при необходимости ``--ocr-version PP-OCRv5`` (см. приложение к документации PaddleOCR).

Пример из корня репозитория::

  python scripts/paddle_image_benchmark.py \\
    --input-dir input/data/1 \\
    --output-dir output/paddle_benchmark \\
    --lang ru \\
    --ocr-version PP-OCRv5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_repo_dotenv() -> None:
    root = _repo_root()
    path = root / ".env"
    if not path.is_file():
        return
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[7:].strip()
        if "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        val = val.strip().strip("'").strip('"')
        os.environ[key] = val


def load_reference(img: Path) -> str | None:
    stem = img.stem
    for name in (f"{stem}.ref.txt", f"{stem}.ref.md", f"{stem}.txt", f"{stem}.md"):
        p = img.parent / name
        if p.is_file():
            t = p.read_text(encoding="utf-8")
            if t.strip():
                return t
    return None


def _inject_metrics_path() -> None:
    p = Path(__file__).resolve().parent / "responses_api_analyze"
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


# Старые/удобные имена → коды PaddleOCR 3.x (см. таблицу lang для PP-OCRv5)
_PADDLE_LANG_ALIASES: dict[str, str] = {
    "cyrillic": "ru",
    "east_slavic": "ru",
    "east-slavic": "ru",
    "russian": "ru",
}


def resolve_paddle_lang(lang: str) -> str:
    key = lang.strip().lower()
    return _PADDLE_LANG_ALIASES.get(key, lang.strip())


def ocr_lines_from_paddle_result(result: object) -> list[str]:
    """Извлекает строки из результата PaddleOCR: 3.x (``predict`` → ``rec_texts``) или 2.x (``ocr``)."""
    lines: list[str] = []
    if not result:
        return lines
    # --- PaddleOCR 3.x / PaddleX: список страниц/результатов с полем rec_texts ---
    if isinstance(result, (list, tuple)) and result:
        first = result[0]
        rec_texts = None
        if isinstance(first, dict):
            rec_texts = first.get("rec_texts")
        else:
            rec_texts = getattr(first, "rec_texts", None)
        if rec_texts is not None:
            for page in result:
                if page is None:
                    continue
                rt = page.get("rec_texts") if isinstance(page, dict) else getattr(page, "rec_texts", None)
                if not rt:
                    continue
                for t in rt:
                    s = str(t).strip()
                    if s:
                        lines.append(s)
            return lines
    # --- PaddleOCR 2.x: result[0] — список [[box], (text, score)], ...] ---
    page = result[0] if isinstance(result, (list, tuple)) and result else result
    if not page:
        return lines
    for item in page:
        if not item or len(item) < 2:
            continue
        txt_part = item[1]
        if isinstance(txt_part, (list, tuple)) and txt_part:
            text = str(txt_part[0])
        elif isinstance(txt_part, str):
            text = txt_part
        else:
            continue
        if text.strip():
            lines.append(text)
    return lines


def run_paddle_ocr(
    *,
    image: Path,
    lang: str,
    ocr_version: str | None,
    use_angle_cls: bool,
    use_gpu: bool | None,
    enable_mkldnn: bool = False,
) -> tuple[str | None, str]:
    """
    Возвращает (ошибка или None, текст гипотезы).

    PaddleOCR **3.x** (PaddleX): в общий конструктор не попадают аргументы 2.x вроде
    ``use_gpu`` / ``show_log`` — устройство через ``device``, логи через ``logging``
    (см. документацию Logging в 3.x). ``use_angle_cls`` пока поддерживается как
    устаревший алиас к ``use_textline_orientation``.

    На CPU у части сборок Paddle включённый OneDNN даёт
    ``ConvertPirAttribute2RuntimeAttribute`` / ``onednn_instruction``; по умолчанию
    передаём ``enable_mkldnn=False`` (см. общие аргументы PaddleX в 3.x).
    """
    try:
        from paddleocr import PaddleOCR  # type: ignore[import-untyped]
    except ImportError as e:
        return f"нет пакета paddleocr (pip install paddlepaddle paddleocr): {e}", ""

    # 3.x: show_log убран; тише консоль через стандартный logging
    for _log_name in ("paddleocr", "paddlex", "paddle"):
        logging.getLogger(_log_name).setLevel(logging.WARNING)

    kwargs: dict = {
        "use_angle_cls": use_angle_cls,
        "lang": lang,
    }
    if ocr_version:
        kwargs["ocr_version"] = ocr_version
    # 3.x: вместо use_gpu
    if use_gpu is True:
        kwargs["device"] = "gpu:0"
    elif use_gpu is False:
        kwargs["device"] = "cpu"

    # PaddleOCR 3.x / PaddleX: общий аргумент пайплайна (не 2.x use_mkldnn в старом API)
    if not enable_mkldnn:
        kwargs["enable_mkldnn"] = False

    ocr = None
    last_err: Exception | None = None
    for _ in range(6):
        try:
            ocr = PaddleOCR(**kwargs)
            break
        except TypeError as e:
            last_err = e
            if "device" in kwargs:
                kwargs.pop("device", None)
                continue
            if "enable_mkldnn" in kwargs:
                kwargs.pop("enable_mkldnn", None)
                continue
            if "ocr_version" in kwargs:
                kwargs.pop("ocr_version", None)
                continue
            return f"PaddleOCR init: {e}", ""
        except Exception as e:
            last_err = e
            err_l = str(e).lower()
            if ("unknown argument" in err_l or "unexpected keyword" in err_l) and "device" in kwargs:
                kwargs.pop("device", None)
                continue
            if ("unknown argument" in err_l or "unexpected keyword" in err_l) and "ocr_version" in kwargs:
                kwargs.pop("ocr_version", None)
                continue
            if ("unknown argument" in err_l or "unexpected keyword" in err_l) and "enable_mkldnn" in kwargs:
                kwargs.pop("enable_mkldnn", None)
                continue
            return f"PaddleOCR init: {e}", ""
    if ocr is None:
        return f"PaddleOCR init: {last_err!s}" if last_err else "PaddleOCR init: неизвестная ошибка", ""

    try:
        # 3.x: ocr() — обёртка над predict(); у predict() нет аргумента cls=.
        if hasattr(ocr, "predict"):
            result = ocr.predict(str(image))
        else:
            result = ocr.ocr(str(image), cls=bool(use_angle_cls))
    except Exception as e:
        return f"PaddleOCR infer: {e}", ""

    lines = ocr_lines_from_paddle_result(result)
    return None, "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=None, help="Каталог с изображениями (по умолчанию input/data/1)")
    p.add_argument("--glob", default="*.png", help='Шаблон файлов, напр. "*.png"')
    p.add_argument("--output-dir", type=Path, default=None, help="Куда писать артефакты (по умолчанию output/paddle_benchmark)")
    p.add_argument(
        "--lang",
        default=os.environ.get("PADDLE_OCR_LANG", "ru"),
        help="Код языка PaddleOCR 3.x (русский: ru). Алиасы: cyrillic→ru (см. доку PaddleOCR PP-OCRv5)",
    )
    p.add_argument(
        "--ocr-version",
        default=os.environ.get("PADDLE_OCR_VERSION") or "PP-OCRv5",
        help="Версия OCR (PaddleOCR 3.x), например PP-OCRv5, PP-OCRv4. Пустая строка = не передавать в API",
    )
    p.add_argument("--no-angle-cls", action="store_true", help="Отключить классификацию угла текста")
    p.add_argument(
        "--use-gpu",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Предпочтение устройства: в PaddleOCR 3.x передаётся как device=gpu:0/cpu (не use_gpu)",
    )
    p.add_argument(
        "--enable-mkldnn",
        action="store_true",
        help="Включить MKL-DNN/OneDNN в PaddleX (по умолчанию выкл. — обход падений CPU вроде onednn_instruction / ConvertPirAttribute2RuntimeAttribute)",
    )
    p.add_argument(
        "--normalize",
        default=os.environ.get("PADDLE_NORMALIZE", "nfkc_ws"),
        help="Режим нормализации для CER (как в mineru_image_benchmark)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Не вызывать PaddleOCR; пересчитать метрики по hypotheses/paddle/<stem>.txt",
    )
    return p.parse_args()


def main() -> int:
    load_repo_dotenv()
    _inject_metrics_path()
    from metrics import build_metric_row, char_error_rate, normalize_text

    args = parse_args()
    inp = args.input_dir or Path(os.environ.get("PADDLE_INPUT_DIR", "input/data/1"))
    if not inp.is_dir():
        raise SystemExit(f"Нет каталога: {inp.resolve()}")

    ocr_ver = (args.ocr_version or "").strip() or None

    out_root = args.output_dir or Path(os.environ.get("PADDLE_OUTPUT_DIR", "output/paddle_benchmark"))
    out_root.mkdir(parents=True, exist_ok=True)
    hyp_root = out_root / "hypotheses" / "paddle"
    hyp_root.mkdir(parents=True, exist_ok=True)

    images = sorted(inp.glob(args.glob))
    if not images:
        print("Нет файлов по шаблону", args.glob, "в", inp)
        return 0

    jsonl_path = out_root / "paddle_runs.jsonl"
    refs_concat: list[str] = []
    hyps_concat: list[str] = []
    elapsed_ok: list[float] = []
    n_files_with_reference = 0

    lang_raw = (args.lang or "ru").strip()
    lang = resolve_paddle_lang(lang_raw)
    if lang.lower() != lang_raw.lower():
        print(f"paddle: язык {lang_raw!r} → {lang!r} (требование PaddleOCR 3.x / PP-OCRv5)")
    model_label = f"paddleocr:lang={lang}" + (f":{ocr_ver}" if ocr_ver else "")
    use_angle_cls = not args.no_angle_cls

    use_gpu: bool | None = args.use_gpu
    if use_gpu is None:
        try:
            import paddle  # type: ignore[import-untyped]

            use_gpu = bool(paddle.device.cuda.device_count() > 0)  # noqa: SLF001
        except Exception:
            use_gpu = False

    env_mkldnn = os.environ.get("PADDLE_ENABLE_MKLDNN", "").strip().lower() in ("1", "true", "yes", "on")
    enable_mkldnn = bool(args.enable_mkldnn or env_mkldnn)

    raw_hypotheses: dict[str, str] = {}

    if args.dry_run:
        print(
            "Режим --dry-run: PaddleOCR не вызывается. Нужны файлы",
            hyp_root / "<stem>.txt",
        )

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for img in images:
            if not img.is_file():
                continue
            ref_raw = load_reference(img)
            had_ref = bool(ref_raw and ref_raw.strip())
            if had_ref:
                n_files_with_reference += 1
            ref_n = normalize_text(ref_raw or "", args.normalize)

            hyp_path = hyp_root / f"{img.stem}.txt"
            t0 = time.perf_counter()
            err: str | None = None
            text = ""

            if args.dry_run:
                if hyp_path.is_file():
                    text = hyp_path.read_text(encoding="utf-8", errors="replace")
                    if text.strip():
                        raw_hypotheses[img.stem] = text
                else:
                    err = "dry-run: нет файла гипотезы " + str(hyp_path)
            else:
                err, text = run_paddle_ocr(
                    image=img,
                    lang=lang,
                    ocr_version=ocr_ver,
                    use_angle_cls=use_angle_cls,
                    use_gpu=use_gpu,
                    enable_mkldnn=enable_mkldnn,
                )
                if err is None and not text.strip():
                    err = "пустой текст OCR"
                elif err is None:
                    hyp_path.write_text(text, encoding="utf-8")
                    raw_hypotheses[img.stem] = text

            elapsed_sec = round(time.perf_counter() - t0, 4) if err is None else None
            if err is None and elapsed_sec is not None:
                elapsed_ok.append(float(elapsed_sec))

            row: dict = {
                "file": img.name,
                "provider": "paddleocr",
                "error": err,
                "elapsed_sec": elapsed_sec,
                "hypothesis_path": str(hyp_path) if err is None else None,
                "meta": {
                    "lang": lang,
                    "lang_cli": lang_raw,
                    "ocr_version": ocr_ver,
                    "use_angle_cls": use_angle_cls,
                    "use_gpu_preference": use_gpu,
                    "enable_mkldnn": enable_mkldnn,
                    "dry_run": args.dry_run,
                },
                "had_reference": had_ref,
                "CER": None,
                "char_accuracy": None,
            }
            if err is None and had_ref:
                cer = char_error_rate(ref_n, normalize_text(text, args.normalize))
                row["CER"] = round(cer, 6)
                row["char_accuracy"] = round(max(0.0, min(1.0, 1.0 - cer)), 6)
                refs_concat.append(ref_n)
                hyps_concat.append(normalize_text(text, args.normalize))
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_cer = len(refs_concat)
    comment = (
        f"paddle_image_benchmark; n_files={len(images)}; n_with_cer={n_cer}; "
        f"lang={lang}; ocr_version={ocr_ver or 'default'}"
    )
    if not refs_concat:
        if args.dry_run:
            token_note = (
                "режим --dry-run: нет hypotheses/paddle/<stem>.txt — сначала полный прогон без --dry-run."
            )
        elif n_files_with_reference > 0:
            token_note = (
                "эталоны есть, но нет успешных гипотез — см. error в paddle_runs.jsonl."
            )
        else:
            token_note = "нет эталонов рядом с PNG — CER не считался"
        summary = {
            "Model": model_label,
            "Final Score": None,
            "Accuracy": None,
            "CER": None,
            "Unit Test Rate": None,
            "Внутренний парсер": "paddleocr",
            "Токены": {"note": token_note},
            "Доп. Комментарии": comment,
            "Скорость": None,
            "_run_mode": {"dry_run": bool(args.dry_run)},
        }
    else:
        summary = build_metric_row(
            ref_raw="\n".join(refs_concat),
            hyp_raw="\n".join(hyps_concat),
            normalize=args.normalize,
            model=model_label,
            internal_parser="paddleocr",
            extra_comment=comment,
            mean_elapsed_sec=statistics.mean(elapsed_ok) if elapsed_ok else None,
        )
        if elapsed_ok:
            summary["Скорость"] = {
                "mean_elapsed_sec_per_file": round(statistics.mean(elapsed_ok), 4),
                "total_elapsed_sec": round(sum(elapsed_ok), 3),
                "n_files_timed": len(elapsed_ok),
            }
        summary["_run_mode"] = {"dry_run": bool(args.dry_run)}

    bundle_json = out_root / "paddle_hypotheses_raw.json"
    bundle_txt = out_root / "paddle_hypotheses_concat.txt"
    if raw_hypotheses:
        bundle_json.write_text(
            json.dumps(raw_hypotheses, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        bundle_txt.write_text(
            "\n\n".join(f"===== {stem} =====\n{text}" for stem, text in sorted(raw_hypotheses.items())),
            encoding="utf-8",
        )

    out_meta: dict[str, object] = {
        "hypotheses_dir": str(hyp_root.resolve()),
        "jsonl": str(jsonl_path.resolve()),
        "summaries_json": str((out_root / "paddle_summaries.json").resolve()),
    }
    if raw_hypotheses:
        out_meta["hypotheses_bundle_json"] = str(bundle_json.resolve())
        out_meta["hypotheses_concat_txt"] = str(bundle_txt.resolve())
        out_meta["n_bundled_files"] = len(raw_hypotheses)
    summary["_outputs"] = out_meta

    (out_root / "paddle_summaries.json").write_text(
        json.dumps({"paddleocr": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Записано:", jsonl_path)
    print("Сводка:", out_root / "paddle_summaries.json")
    print("Гипотезы (текст):", hyp_root)
    if raw_hypotheses:
        print("Сырой текст (JSON):", bundle_json.resolve())
        print("Сырой текст (склейка):", bundle_txt.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
