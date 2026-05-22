#!/usr/bin/env python3
"""
Прогон **GOT-OCR2.0** (Hugging Face ``ucaslcl/GOT-OCR2_0``) по изображениям и метрики против эталонов,
в том же формате, что ``mineru_image_benchmark`` / ``paddle_image_benchmark``.

Пишет плоский текст в ``hypotheses/got/<stem>.txt``, JSONL ``got_runs.jsonl``,
сводку ``got_summaries.json`` (ключ **``got_ocr2``**), плюс ``got_hypotheses_raw.json`` /
``got_hypotheses_concat.txt`` и ``got_ocr2._outputs``.

**Эталоны:** ``<stem>.ref.txt`` → ``<stem>.ref.md`` → ``<stem>.txt`` → ``<stem>.md``.

**Зависимости:** ``pip install jiwer`` + зависимости модели (см. ``notes/requirements-ocr-notebook-colab.txt``).
Важно: remote code **GOT-OCR2_0** совместим с **transformers 4.40.x** (на 4.41+ часто
``DynamicCache`` / ``seen_tokens``). Ставьте зависимости из репозитория или
``pip install 'transformers>=4.40.0,<4.41.0'``.

Пример из корня репозитория::

  python scripts/got_image_benchmark.py \\
    --input-dir input/data/1 \\
    --output-dir output/got_benchmark
"""

from __future__ import annotations

import argparse
import json
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


def pick_torch_dtype():
    import torch

    if not torch.cuda.is_available():
        return torch.float32, "cpu"
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        return torch.bfloat16, "cuda"
    return torch.float16, "cuda"


def load_got_model(model_id: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dtype, device_str = pick_torch_dtype()
    load_kw: dict = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    if device_str == "cuda":
        load_kw["device_map"] = "cuda"
    model = AutoModel.from_pretrained(model_id, **load_kw)
    model.eval()
    if device_str == "cpu":
        model = model.to(torch.float32)
    return model, tokenizer, device_str, dtype


def run_got_chat(model, tokenizer, image: Path, ocr_type: str) -> tuple[str | None, str, float]:
    t0 = time.perf_counter()
    try:
        res = model.chat(tokenizer, str(image), ocr_type=ocr_type)
        hyp = res if isinstance(res, str) else str(res)
        return None, hyp, time.perf_counter() - t0
    except Exception as e:
        msg = str(e)
        if "seen_tokens" in msg or "DynamicCache" in msg:
            msg += (
                " — типичная несовместимость с transformers>=4.41; "
                "pip install 'transformers>=4.40.0,<4.41.0' (см. notes/requirements-ocr-notebook-colab.txt)"
            )
        return f"GOT infer: {msg}", "", time.perf_counter() - t0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=None, help="Каталог с изображениями (по умолчанию input/data/1)")
    p.add_argument("--glob", default="*.png", help='Шаблон файлов, напр. "*.png"')
    p.add_argument("--output-dir", type=Path, default=None, help="Куда писать артефакты (по умолчанию output/got_benchmark)")
    p.add_argument(
        "--model-id",
        default=os.environ.get("GOT_MODEL_ID", "ucaslcl/GOT-OCR2_0"),
        help="Идентификатор модели на Hugging Face (remote code)",
    )
    p.add_argument(
        "--ocr-type",
        default=os.environ.get("GOT_OCR_TYPE", "ocr"),
        help='Режим model.chat(..., ocr_type=...): обычно "ocr" или "format" (см. README модели)',
    )
    p.add_argument(
        "--normalize",
        default=os.environ.get("GOT_NORMALIZE", "nfkc_ws"),
        help="Режим нормализации для CER (как в mineru_image_benchmark)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Не загружать GOT и не вызывать inference; пересчитать метрики по hypotheses/got/<stem>.txt",
    )
    return p.parse_args()


def main() -> int:
    load_repo_dotenv()
    _inject_metrics_path()
    from metrics import build_metric_row, char_error_rate, normalize_text

    args = parse_args()
    inp = args.input_dir or Path(os.environ.get("GOT_INPUT_DIR", "input/data/1"))
    if not inp.is_dir():
        raise SystemExit(f"Нет каталога: {inp.resolve()}")

    out_root = args.output_dir or Path(os.environ.get("GOT_OUTPUT_DIR", "output/got_benchmark"))
    out_root.mkdir(parents=True, exist_ok=True)
    hyp_root = out_root / "hypotheses" / "got"
    hyp_root.mkdir(parents=True, exist_ok=True)

    images = sorted(inp.glob(args.glob))
    if not images:
        print("Нет файлов по шаблону", args.glob, "в", inp)
        return 0

    jsonl_path = out_root / "got_runs.jsonl"
    refs_concat: list[str] = []
    hyps_concat: list[str] = []
    elapsed_ok: list[float] = []
    n_files_with_reference = 0

    model_label = f"got:{args.model_id}|ocr_type={args.ocr_type}"
    raw_hypotheses: dict[str, str] = {}

    model = tokenizer = None
    device_str = dtype_s = None
    if not args.dry_run:
        try:
            model, tokenizer, device_str, dtype_s = load_got_model(args.model_id)
        except ImportError as e:
            raise SystemExit(
                f"Нет зависимостей для GOT (pip install -r notes/requirements-ocr-notebook-colab.txt): {e}"
            ) from e
        except Exception as e:
            raise SystemExit(f"Не удалось загрузить модель {args.model_id!r}: {e}") from e
        print("GOT модель загружена:", args.model_id, "| device =", device_str, "| dtype =", dtype_s)
    else:
        print(
            "Режим --dry-run: GOT не загружается. Нужны файлы",
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
                assert model is not None and tokenizer is not None
                err, text, _infer_sec = run_got_chat(model, tokenizer, img, args.ocr_type)
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
                "provider": "got_ocr2",
                "error": err,
                "elapsed_sec": elapsed_sec if err is None else None,
                "hypothesis_path": str(hyp_path) if err is None else None,
                "meta": {
                    "model_id": args.model_id,
                    "ocr_type": args.ocr_type,
                    "device": device_str,
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
        f"got_image_benchmark; n_files={len(images)}; n_with_cer={n_cer}; "
        f"model_id={args.model_id}; ocr_type={args.ocr_type}"
    )
    if not refs_concat:
        if args.dry_run:
            token_note = (
                "режим --dry-run: нет hypotheses/got/<stem>.txt — сначала полный прогон без --dry-run."
            )
        elif n_files_with_reference > 0:
            token_note = "эталоны есть, но нет успешных гипотез — см. error в got_runs.jsonl."
        else:
            token_note = "нет эталонов рядом с PNG — CER не считался"
        summary = {
            "Model": model_label,
            "Final Score": None,
            "Accuracy": None,
            "CER": None,
            "Unit Test Rate": None,
            "Внутренний парсер": "got_ocr2",
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
            internal_parser="got_ocr2",
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

    bundle_json = out_root / "got_hypotheses_raw.json"
    bundle_txt = out_root / "got_hypotheses_concat.txt"
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
        "summaries_json": str((out_root / "got_summaries.json").resolve()),
    }
    if raw_hypotheses:
        out_meta["hypotheses_bundle_json"] = str(bundle_json.resolve())
        out_meta["hypotheses_concat_txt"] = str(bundle_txt.resolve())
        out_meta["n_bundled_files"] = len(raw_hypotheses)
    summary["_outputs"] = out_meta

    (out_root / "got_summaries.json").write_text(
        json.dumps({"got_ocr2": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Записано:", jsonl_path)
    print("Сводка:", out_root / "got_summaries.json")
    print("Гипотезы (текст):", hyp_root)
    if raw_hypotheses:
        print("Сырой текст (JSON):", bundle_json.resolve())
        print("Сырой текст (склейка):", bundle_txt.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
