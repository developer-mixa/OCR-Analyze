#!/usr/bin/env python3
"""
Пересчёт **Final Score** (и полей ``build_metric_row``: Accuracy, CER, ``_diagnostics``)
по **уже сохранённым** гипотезам и ``*_runs.jsonl`` — без повторного запуска OCR.

Источники как у ``fill_ocr_analyze_odt.py``:
``output/mineru/*.md``, ``output/paddle/*.txt``, ``output/got/*.txt`` (или ``got_benchmark``),
``output/yandex_vision`` (плоские ``.txt`` или ``hypotheses/yandex_vision``).

Перезаписывает ``mineru_summaries.json``, ``paddle_summaries.json``, ``got_summaries.json``,
``yandex_vision_summaries.json`` в соответствующих каталогах ``output/…`` (сохраняет
``Скорость``, ``_run_mode`` из старого файла при наличии; обновляет ``_outputs`` на текущие пути).

Пример::

  python scripts/recalculate_summaries_from_outputs.py
  python scripts/recalculate_summaries_from_outputs.py --fill-odt
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_ref(inp: Path, stem: str) -> str:
    for name in (f"{stem}.ref.txt", f"{stem}.ref.md", f"{stem}.txt", f"{stem}.md"):
        p = inp / name
        if p.is_file():
            t = p.read_text(encoding="utf-8")
            if t.strip():
                return t
    return ""


def load_jsonl_mean_sec(jsonl: Path) -> float | None:
    if not jsonl.is_file():
        return None
    vals: list[float] = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        t = r.get("elapsed_sec")
        if t is not None:
            vals.append(float(t))
    return statistics.mean(vals) if vals else None


def load_got_hyp(repo: Path, stem: str) -> str:
    for p in (
        repo / "output/got" / f"{stem}.txt",
        repo / "output/got_benchmark/hypotheses/got" / f"{stem}.txt",
    ):
        if p.is_file():
            return p.read_text(encoding="utf-8", errors="replace")
    return ""


def load_yandex_hyp(repo: Path, stem: str) -> str:
    for p in (
        repo / "output/yandex_vision" / f"{stem}.txt",
        repo / "output/yandex_vision/hypotheses/yandex_vision" / f"{stem}.txt",
        repo / "output/yandex_vision_benchmark/hypotheses/yandex_vision" / f"{stem}.txt",
    ):
        if p.is_file():
            return p.read_text(encoding="utf-8", errors="replace")
    return ""


def _inject_metrics() -> None:
    p = _repo_root() / "scripts/responses_api_analyze"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def recalc_one(
    *,
    repo: Path,
    norm: str,
    refs_hyps: tuple[list[str], list[str]],
    summary_path: Path,
    jsonl_path: Path,
    top_key: str,
    build_kw: dict,
    hypotheses_dir: Path,
) -> bool:
    refs, hyps = refs_hyps
    if not refs or not hyps:
        print(f"skip {summary_path.name}: нет пар ref+hyp ({len(refs)=} {len(hyps)=})")
        return False
    _inject_metrics()
    from metrics import build_metric_row

    mean_e = load_jsonl_mean_sec(jsonl_path)
    row = build_metric_row(
        ref_raw="\n".join(refs),
        hyp_raw="\n".join(hyps),
        normalize=norm,
        mean_elapsed_sec=mean_e,
        **build_kw,
    )
    old: dict | None = None
    if summary_path.is_file():
        try:
            old = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old = None
    inner_old = (old or {}).get(top_key) if isinstance(old, dict) else None
    if isinstance(inner_old, dict):
        if inner_old.get("Скорость"):
            row["Скорость"] = inner_old["Скорость"]
        if inner_old.get("_run_mode") is not None:
            row["_run_mode"] = inner_old["_run_mode"]
    out_root = summary_path.parent
    row["_outputs"] = {
        "hypotheses_dir": str(hypotheses_dir.resolve()),
        "jsonl": str(jsonl_path.resolve()),
        "summaries_json": str(summary_path.resolve()),
    }
    for bundle in sorted(out_root.glob("*_hypotheses_raw.json")):
        row["_outputs"]["hypotheses_bundle_json"] = str(bundle.resolve())
        break
    for concat in sorted(out_root.glob("*_hypotheses_concat.txt")):
        row["_outputs"]["hypotheses_concat_txt"] = str(concat.resolve())
        break

    summary_path.write_text(json.dumps({top_key: row}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK", summary_path.relative_to(repo))
    return True


def collect_pairs(
    repo: Path,
    inp: Path,
    stems: list[str],
    hyp_loader,
) -> tuple[list[str], list[str]]:
    refs: list[str] = []
    hyps: list[str] = []

    for stem in stems:
        ref_raw = load_ref(inp, stem)
        hyp_raw = hyp_loader(repo, stem)
        if ref_raw and hyp_raw.strip():
            refs.append(ref_raw)
            hyps.append(hyp_raw)
    return refs, hyps


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", type=Path, default=None, help="Корень репозитория (по умолчанию родитель scripts/)")
    p.add_argument(
        "--fill-odt",
        action="store_true",
        help="После сводок запустить scripts/fill_ocr_analyze_odt.py",
    )
    args = p.parse_args()
    repo = (args.repo or _repo_root()).resolve()
    inp = repo / "input/data/1"
    stems = ["test1", "test2", "test3", "test4"]
    norm = "nfkc_ws"

    def hyp_mineru(r: Path, stem: str) -> str:
        pth = r / "output/mineru" / f"{stem}.md"
        return pth.read_text(encoding="utf-8", errors="replace") if pth.is_file() else ""

    def hyp_paddle(r: Path, stem: str) -> str:
        pth = r / "output/paddle" / f"{stem}.txt"
        return pth.read_text(encoding="utf-8", errors="replace") if pth.is_file() else ""

    recalc_one(
        repo=repo,
        norm=norm,
        refs_hyps=collect_pairs(repo, inp, stems, hyp_mineru),
        summary_path=repo / "output/mineru/mineru_summaries.json",
        jsonl_path=repo / "output/mineru/mineru_runs.jsonl",
        top_key="mineru",
        build_kw=dict(
            model="mineru_cli:pipeline",
            internal_parser="mineru",
            extra_comment="output/mineru/*.md (recalculate_summaries_from_outputs)",
        ),
        hypotheses_dir=repo / "output/mineru",
    )
    recalc_one(
        repo=repo,
        norm=norm,
        refs_hyps=collect_pairs(repo, inp, stems, hyp_paddle),
        summary_path=repo / "output/paddle/paddle_summaries.json",
        jsonl_path=repo / "output/paddle/paddle_runs.jsonl",
        top_key="paddleocr",
        build_kw=dict(
            model="paddleocr:lang=ru:PP-OCRv5",
            internal_parser="paddleocr",
            extra_comment="output/paddle/*.txt (recalculate_summaries_from_outputs)",
        ),
        hypotheses_dir=repo / "output/paddle",
    )
    recalc_one(
        repo=repo,
        norm=norm,
        refs_hyps=collect_pairs(repo, inp, stems, load_got_hyp),
        summary_path=repo / "output/got/got_summaries.json",
        jsonl_path=repo / "output/got/got_runs.jsonl",
        top_key="got_ocr2",
        build_kw=dict(
            model="got:ucaslcl/GOT-OCR2_0|ocr_type=ocr",
            internal_parser="got_ocr2",
            extra_comment="output/got/*.txt (recalculate_summaries_from_outputs)",
        ),
        hypotheses_dir=repo / "output/got",
    )
    ydir = repo / "output/yandex_vision/hypotheses/yandex_vision"
    if not ydir.is_dir():
        ydir = repo / "output/yandex_vision"
    recalc_one(
        repo=repo,
        norm=norm,
        refs_hyps=collect_pairs(repo, inp, stems, load_yandex_hyp),
        summary_path=repo / "output/yandex_vision/yandex_vision_summaries.json",
        jsonl_path=repo / "output/yandex_vision/yandex_vision_runs.jsonl",
        top_key="yandex_vision",
        build_kw=dict(
            model="yandex_vision:page",
            internal_parser="yandex_vision_rest",
            extra_comment="output/yandex_vision (recalculate_summaries_from_outputs)",
        ),
        hypotheses_dir=ydir if ydir.is_dir() else repo / "output/yandex_vision",
    )

    if args.fill_odt:
        fill = repo / "scripts/fill_ocr_analyze_odt.py"
        rc = subprocess.call([sys.executable, str(fill)], cwd=str(repo))
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
