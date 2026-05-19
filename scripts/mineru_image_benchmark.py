#!/usr/bin/env python3
"""
Прогон **MinerU** по изображениям (PNG/JPEG/…) и метрики против эталонов рядом с файлами.

Вызывает CLI ``mineru`` (как в документации MinerU 3.x: без ``--api-url`` поднимается временный
``mineru-api``). Результат извлекается из сгенерированных ``*.md`` в каталоге вывода задачи.

**Эталоны** (как в ``run_ocr_api_benchmark``): ``<stem>.ref.txt`` → ``<stem>.ref.md`` → ``<stem>.txt`` → ``<stem>.md``.

**Зависимости (минимум для метрик):**
  pip install jiwer

**MinerU** — по официальной инструкции, например:
  pip install uv && uv pip install -U "mineru[all]"
  # модели: см. https://opendatalab.github.io/MinerU/

**Google Colab (бесплатный GPU):**
  Runtime → Change runtime type → GPU (T4).
  Если Hugging Face недоступен: ``export MINERU_MODEL_SOURCE=modelscope``
  Для экономии VRAM на T4 разумно: ``--backend pipeline`` (см. https://opendatalab.github.io/MinerU/usage/cli_tools/).

Пример из корня репозитория:
  python scripts/mineru_image_benchmark.py \\
    --input-dir input/data/1 \\
    --output-dir output/mineru_benchmark \\
    --backend pipeline \\
    --lang cyrillic
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
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


def collect_markdown_from_mineru_output(out_dir: Path, image_stem: str) -> str:
    """
    Ищет основной markdown среди артефактов MinerU.
    Предпочтение: файл с именем ``{stem}.md``, иначе самый большой ``*.md`` под out_dir.
    """
    all_md = sorted(out_dir.rglob("*.md"))
    if not all_md:
        return ""
    stem_md = [p for p in all_md if p.stem == image_stem]
    if len(stem_md) == 1:
        return stem_md[0].read_text(encoding="utf-8", errors="replace")
    if stem_md:
        stem_md.sort(key=lambda p: p.stat().st_size, reverse=True)
        return stem_md[0].read_text(encoding="utf-8", errors="replace")
    all_md.sort(key=lambda p: p.stat().st_size, reverse=True)
    return all_md[0].read_text(encoding="utf-8", errors="replace")


def run_mineru_cli(
    *,
    mineru_bin: str,
    image: Path,
    out_dir: Path,
    backend: str,
    method: str | None,
    lang: str | None,
    api_url: str | None,
    timeout_sec: float,
) -> tuple[str | None, str | None]:
    """
    Запускает mineru. Возвращает (stderr+stdout при ошибке, None) или (None, None) при успехе.
    """
    cmd: list[str] = [mineru_bin, "-p", str(image), "-o", str(out_dir), "-b", backend]
    if method:
        cmd.extend(["-m", method])
    if lang:
        cmd.extend(["-l", lang])
    if api_url:
        cmd.extend(["--api-url", api_url])
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return f"timeout after {timeout_sec}s", None
    except FileNotFoundError:
        return f"не найден исполняемый файл: {mineru_bin!r}", None
    if proc.returncode != 0:
        tail = (proc.stderr or "") + "\n" + (proc.stdout or "")
        return tail.strip()[-8000:] or f"exit code {proc.returncode}", None
    return None, None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=None, help="Каталог с изображениями (по умолчанию input/data/1)")
    p.add_argument("--glob", default="*.png", help='Шаблон файлов, напр. "*.png"')
    p.add_argument("--output-dir", type=Path, default=None, help="Куда писать JSONL, summaries, hypotheses (по умолчанию output/mineru_benchmark)")
    p.add_argument(
        "--mineru-bin",
        default=os.environ.get("MINERU_BIN", "mineru"),
        help="Путь к CLI mineru (по умолчанию $MINERU_BIN или mineru в PATH)",
    )
    p.add_argument(
        "--backend",
        default=os.environ.get("MINERU_BACKEND", "pipeline"),
        help="Значение -b: pipeline | hybrid-auto-engine | … (см. mineru --help). Для Colab T4 обычно pipeline",
    )
    p.add_argument(
        "--method",
        default=os.environ.get("MINERU_METHOD") or None,
        help="Опционально -m: auto | txt | ocr (для сканов часто ocr)",
    )
    p.add_argument(
        "--lang",
        default=os.environ.get("MINERU_LANG") or None,
        help="Опционально -l: для русского текста часто cyrillic или east_slavic (см. mineru --help)",
    )
    p.add_argument(
        "--api-url",
        default=os.environ.get("MINERU_API_URL") or None,
        help="Если задан, mineru ходит в существующий mineru-api (--api-url), иначе локальный временный API",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("MINERU_FILE_TIMEOUT_SEC", "3600")),
        help="Таймаут на один файл (секунды)",
    )
    p.add_argument(
        "--normalize",
        default=os.environ.get("MINERU_NORMALIZE", "nfkc_ws"),
        help="Режим нормализации для CER (как в api benchmark)",
    )
    p.add_argument(
        "--keep-work",
        action="store_true",
        help="Не удалять рабочие каталоги _work/<stem> после успешного прогона",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Не вызывать mineru; только пересчитать метрики, если в hypotheses уже есть .md",
    )
    return p.parse_args()


def main() -> int:
    load_repo_dotenv()
    _inject_metrics_path()
    from metrics import build_metric_row, char_error_rate, normalize_text

    args = parse_args()
    inp = args.input_dir or Path(os.environ.get("MINERU_INPUT_DIR", "input/data/1"))
    if not inp.is_dir():
        raise SystemExit(f"Нет каталога: {inp.resolve()}")

    out_root = args.output_dir or Path(os.environ.get("MINERU_OUTPUT_DIR", "output/mineru_benchmark"))
    out_root.mkdir(parents=True, exist_ok=True)
    hyp_root = out_root / "hypotheses" / "mineru"
    hyp_root.mkdir(parents=True, exist_ok=True)
    work_root = out_root / "_mineru_work"
    work_root.mkdir(parents=True, exist_ok=True)

    images = sorted(inp.glob(args.glob))
    if not images:
        print("Нет файлов по шаблону", args.glob, "в", inp)
        return 0

    jsonl_path = out_root / "mineru_runs.jsonl"
    refs_concat: list[str] = []
    hyps_concat: list[str] = []
    elapsed_ok: list[float] = []

    backend = args.backend.strip()
    model_label = f"mineru_cli:{backend}"
    if args.api_url:
        model_label += f"(api={args.api_url})"

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for img in images:
            if not img.is_file():
                continue
            ref_raw = load_reference(img)
            had_ref = bool(ref_raw and ref_raw.strip())
            ref_n = normalize_text(ref_raw or "", args.normalize)

            hyp_path = hyp_root / f"{img.stem}.md"
            t0 = time.perf_counter()
            err: str | None = None
            text = ""

            if args.dry_run:
                if hyp_path.is_file():
                    text = hyp_path.read_text(encoding="utf-8", errors="replace")
                else:
                    err = "dry-run: нет файла гипотезы " + str(hyp_path)
            else:
                wdir = work_root / img.stem
                if wdir.exists():
                    shutil.rmtree(wdir, ignore_errors=True)
                wdir.mkdir(parents=True, exist_ok=True)
                cli_err, _ = run_mineru_cli(
                    mineru_bin=args.mineru_bin,
                    image=img,
                    out_dir=wdir,
                    backend=backend,
                    method=args.method,
                    lang=args.lang,
                    api_url=args.api_url,
                    timeout_sec=args.timeout,
                )
                if cli_err:
                    err = cli_err
                else:
                    text = collect_markdown_from_mineru_output(wdir, img.stem)
                    if not text.strip():
                        err = "mineru завершился без .md или пустой markdown"
                    else:
                        hyp_path.write_text(text, encoding="utf-8")
                        if not args.keep_work:
                            shutil.rmtree(wdir, ignore_errors=True)

            elapsed_sec = round(time.perf_counter() - t0, 4) if err is None else None
            if err is None and elapsed_sec is not None:
                elapsed_ok.append(float(elapsed_sec))

            row: dict = {
                "file": img.name,
                "provider": "mineru",
                "error": err,
                "elapsed_sec": elapsed_sec,
                "hypothesis_path": str(hyp_path) if err is None else None,
                "meta": {
                    "backend": backend,
                    "method": args.method,
                    "lang": args.lang,
                    "mineru_bin": args.mineru_bin,
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
    comment = f"mineru_image_benchmark; n_files={len(images)}; n_with_cer={n_cer}; backend={backend}"
    if not refs_concat:
        summary = {
            "Model": model_label,
            "Final Score": None,
            "Accuracy": None,
            "CER": None,
            "Unit Test Rate": None,
            "Внутренний парсер": "mineru",
            "Токены": {"note": "нет эталонов — CER не считался"},
            "Доп. Комментарии": comment,
            "Скорость": None,
        }
    else:
        summary = build_metric_row(
            ref_raw="\n".join(refs_concat),
            hyp_raw="\n".join(hyps_concat),
            normalize=args.normalize,
            model=model_label,
            internal_parser="mineru",
            extra_comment=comment,
        )
        if elapsed_ok:
            summary["Скорость"] = {
                "mean_elapsed_sec_per_file": round(statistics.mean(elapsed_ok), 4),
                "total_elapsed_sec": round(sum(elapsed_ok), 3),
                "n_files_timed": len(elapsed_ok),
            }

    (out_root / "mineru_summaries.json").write_text(
        json.dumps({"mineru": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Записано:", jsonl_path)
    print("Сводка:", out_root / "mineru_summaries.json")
    print("Гипотезы:", hyp_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
