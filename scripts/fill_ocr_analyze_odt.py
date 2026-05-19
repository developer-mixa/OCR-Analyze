#!/usr/bin/env python3
"""
Заполняет ocr-analyze.odt: сводные метрики MinerU, Paddle, GOT и таблица по файлам.
Источники: input/data/1/*.ref.txt, output/mineru/*.md, output/paddle/*.txt, output/got/*.txt
(или hypotheses/got в output/got_benchmark), jsonl с elapsed в output/{mineru,paddle,got}/.
"""
from __future__ import annotations

import io
import json
import statistics
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


def xml_text(s: str) -> str:
    return escape(str(s), {'"': "&quot;", "'": "&apos;", "&": "&amp;", "<": "&lt;"})


def cell(style: str, pstyle: str, text: str) -> str:
    return (
        f'<table:table-cell table:style-name="{style}" office:value-type="string">'
        f'<text:p text:style-name="{pstyle}">{xml_text(text)}</text:p>'
        "</table:table-cell>"
    )


def header_row(labels: list[str]) -> str:
    parts = ["<table:table-row>"]
    for i, lab in enumerate(labels):
        st = "Таблица1.F1" if i == len(labels) - 1 else "Таблица1.A1"
        parts.append(cell(st, "P4", lab))
    parts.append("</table:table-row>")
    return "".join(parts)


def data_row(vals: list[str], *, numeric_middle: bool = True) -> str:
    parts = ["<table:table-row>"]
    n = len(vals)
    for i, v in enumerate(vals):
        st = "Таблица1.F2" if i == n - 1 else "Таблица1.A2"
        if numeric_middle and n > 2:
            pstyle = "P5" if 0 < i < n - 1 else "P4"
        else:
            pstyle = "P4"
        parts.append(cell(st, pstyle, v))
    parts.append("</table:table-row>")
    return "".join(parts)


def build_table(name: str, headers: list[str], rows: list[list[str]]) -> str:
    out = [
        f'<table:table table:name="{xml_text(name)}" table:style-name="Таблица1">',
        '<table:table-column table:style-name="Таблица1.A" table:number-columns-repeated="'
        f'{len(headers)}"/>',
        header_row(headers),
    ]
    for r in rows:
        out.append(data_row(r))
    out.append("</table:table>")
    return "".join(out)


def load_got_hyp_raw(repo: Path, stem: str) -> str:
    for p in (
        repo / "output" / "got" / f"{stem}.txt",
        repo / "output" / "got_benchmark" / "hypotheses" / "got" / f"{stem}.txt",
    ):
        if p.is_file():
            return p.read_text(encoding="utf-8", errors="replace")
    return ""


def got_jsonl_path(repo: Path) -> Path | None:
    for p in (repo / "output" / "got" / "got_runs.jsonl", repo / "output" / "got_benchmark" / "got_runs.jsonl"):
        if p.is_file():
            return p
    return None


def best_by_cer(names_cers: list[tuple[str, float]]) -> str:
    min_c = min(c for _, c in names_cers)
    names = [n for n, c in names_cers if abs(c - min_c) < 1e-9]
    return ", ".join(names) if len(names) > 1 else names[0]


def load_jsonl_elapsed(path: Path) -> dict[str, float]:
    """stem (без .png) -> elapsed_sec из jsonl бенчмарка."""
    if not path.is_file():
        return {}
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        stem = Path(r.get("file", "")).stem
        t = r.get("elapsed_sec")
        if stem and t is not None:
            out[stem] = float(t)
    return out


def find_report_block(content: str) -> tuple[int, int] | None:
    """Диапазон в content.xml: от абзаца методики / первой сводной таблицы до <text:p P3/>."""
    key = "Методика: эталоны input/data/1"
    if key in content:
        i = content.index(key)
        start = content.rfind("<text:p ", 0, i)
        end = content.find('<text:p text:style-name="P3"/>', i)
        if end == -1:
            end = content.find("</office:text>", i)
        if start == -1 or end == -1 or start >= end:
            return None
        return start, end
    for tn in ("ТаблицаСводка", "Таблица1"):
        tag = f'table:name="{tn}"'
        j = content.find(tag)
        if j == -1:
            continue
        start = content.rfind("<table:table", 0, j + 1)
        pos = start
        for _ in range(2):
            k = content.find("</table:table>", pos)
            if k == -1:
                return None
            pos = k + len("</table:table>")
        end = content.find('<text:p text:style-name="P3"/>', start)
        if end == -1:
            end = pos
        return start, end
    return None


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "scripts/responses_api_analyze"))
    from metrics import build_metric_row, char_error_rate, normalize_text

    import jiwer

    inp = repo / "input/data/1"
    norm = "nfkc_ws"
    stems = ["test1", "test2", "test3", "test4"]

    def load_ref(stem: str) -> str:
        for name in (f"{stem}.ref.txt", f"{stem}.ref.md", f"{stem}.txt", f"{stem}.md"):
            p = inp / name
            if p.is_file():
                t = p.read_text(encoding="utf-8")
                if t.strip():
                    return t
        return ""

    refs_m: list[str] = []
    hyps_m: list[str] = []
    per_m: list[tuple[str, float, float]] = []
    for stem in stems:
        ref_raw = load_ref(stem)
        md = repo / "output/mineru" / f"{stem}.md"
        hyp_raw = md.read_text(encoding="utf-8") if md.is_file() else ""
        ref_n = normalize_text(ref_raw, norm)
        hyp_n = normalize_text(hyp_raw, norm)
        cer = char_error_rate(ref_n, hyp_n) if ref_raw else 1.0
        wer = float(jiwer.wer(ref_n, hyp_n)) if ref_n else 1.0
        per_m.append((stem, cer, wer))
        if ref_raw and hyp_raw.strip():
            refs_m.append(ref_n)
            hyps_m.append(hyp_n)

    row_m = build_metric_row(
        ref_raw="\n".join(refs_m),
        hyp_raw="\n".join(hyps_m),
        normalize=norm,
        model="mineru_cli:pipeline",
        internal_parser="mineru",
        extra_comment="output/mineru/*.md",
    )
    wer_m = row_m["_diagnostics"]["WER"]

    refs_p: list[str] = []
    hyps_p: list[str] = []
    per_p: list[tuple[str, float, float]] = []
    for stem in stems:
        ref_raw = load_ref(stem)
        tx = repo / "output/paddle" / f"{stem}.txt"
        hyp_raw = tx.read_text(encoding="utf-8") if tx.is_file() else ""
        ref_n = normalize_text(ref_raw, norm)
        hyp_n = normalize_text(hyp_raw, norm)
        cer = char_error_rate(ref_n, hyp_n)
        wer = float(jiwer.wer(ref_n, hyp_n))
        per_p.append((stem, cer, wer))
        refs_p.append(ref_n)
        hyps_p.append(hyp_n)

    row_p = build_metric_row(
        ref_raw="\n".join(refs_p),
        hyp_raw="\n".join(hyps_p),
        normalize=norm,
        model="paddleocr:lang=ru:PP-OCRv5",
        internal_parser="paddleocr",
        extra_comment="output/paddle/*.txt",
    )
    wer_p = row_p["_diagnostics"]["WER"]

    refs_g: list[str] = []
    hyps_g: list[str] = []
    per_g: list[tuple[str, float | None, float | None]] = []
    for stem in stems:
        ref_raw = load_ref(stem)
        hyp_raw = load_got_hyp_raw(repo, stem)
        ref_n = normalize_text(ref_raw, norm)
        hyp_n = normalize_text(hyp_raw, norm)
        if ref_raw and hyp_raw.strip():
            cer = char_error_rate(ref_n, hyp_n)
            wer = float(jiwer.wer(ref_n, hyp_n))
            refs_g.append(ref_n)
            hyps_g.append(hyp_n)
        else:
            cer = wer = None
        per_g.append((stem, cer, wer))

    row_g: dict | None = None
    wer_g: float | None = None
    if refs_g and hyps_g:
        row_g = build_metric_row(
            ref_raw="\n".join(refs_g),
            hyp_raw="\n".join(hyps_g),
            normalize=norm,
            model="got:ucaslcl/GOT-OCR2_0|ocr_type=ocr",
            internal_parser="got_ocr2",
            extra_comment="output/got/*.txt",
        )
        wer_g = float(row_g["_diagnostics"]["WER"])

    mineru_times = load_jsonl_elapsed(repo / "output/mineru/mineru_runs.jsonl")
    paddle_times = load_jsonl_elapsed(repo / "output/paddle/paddle_runs.jsonl")
    got_jsonl = got_jsonl_path(repo)
    got_times = load_jsonl_elapsed(got_jsonl) if got_jsonl else {}
    mean_m = statistics.mean(mineru_times.values()) if mineru_times else None
    mean_p = statistics.mean(paddle_times.values()) if paddle_times else None
    mean_g = statistics.mean(got_times.values()) if got_times else None

    mineru_speed = f"{mean_m:.2f} с/файл" if mean_m is not None else "н/д (нет mineru_runs.jsonl)"
    paddle_speed = f"{mean_p:.2f} с/файл" if mean_p is not None else "н/д (нет paddle_runs.jsonl)"
    got_speed = f"{mean_g:.2f} с/файл" if mean_g is not None else "н/д (нет got_runs.jsonl)"

    summary_headers = ["Model", "Final Score", "Accuracy", "CER", "WER", "Avg speed"]
    summary_rows = [
        [
            "MinerU (mineru_cli:pipeline)",
            f"{row_m['Final Score']}",
            f"{row_m['Accuracy']}",
            f"{row_m['CER']}",
            f"{round(wer_m, 6)}",
            mineru_speed,
        ],
        [
            "PaddleOCR (ru, PP-OCRv5, CPU)",
            f"{row_p['Final Score']}",
            f"{row_p['Accuracy']}",
            f"{row_p['CER']}",
            f"{round(wer_p, 6)}",
            paddle_speed,
        ],
    ]
    if row_g is not None and wer_g is not None:
        summary_rows.append(
            [
                "GOT-OCR2.0 (HF, ocr)",
                f"{row_g['Final Score']}",
                f"{row_g['Accuracy']}",
                f"{row_g['CER']}",
                f"{round(wer_g, 6)}",
                got_speed,
            ]
        )

    per_headers = [
        "Файл",
        "MinerU CER",
        "MU WER",
        "MU,с",
        "Paddle CER",
        "PD WER",
        "PD,с",
        "GOT CER",
        "GOT WER",
        "GOT,с",
        "Лучший CER",
    ]
    per_rows: list[list[str]] = []
    for (sm, cm, wm), (sp, cp, wp), (sg, cg, wg) in zip(per_m, per_p, per_g, strict=True):
        assert sm == sp == sg
        img = f"{sm}.png"
        tm = mineru_times.get(sm)
        tp = paddle_times.get(sm)
        tg = got_times.get(sm)
        parts_cer: list[tuple[str, float]] = [("MinerU", cm), ("Paddle", cp)]
        if cg is not None:
            parts_cer.append(("GOT", cg))
        winner = best_by_cer(parts_cer)
        per_rows.append(
            [
                img,
                f"{cm:.6f}",
                f"{wm:.6f}",
                f"{tm:.4f}" if tm is not None else "—",
                f"{cp:.6f}",
                f"{wp:.6f}",
                f"{tp:.4f}" if tp is not None else "—",
                f"{cg:.6f}" if cg is not None else "—",
                f"{wg:.6f}" if wg is not None else "—",
                f"{tg:.4f}" if tg is not None else "—",
                winner,
            ]
        )

    table_summary = build_table("ТаблицаСводка", summary_headers, summary_rows)
    table_per = build_table("ТаблицаПоФайлам", per_headers, per_rows)

    note = (
        "Методика: эталоны input/data/1 (*.ref.txt), нормализация nfkc_ws, CER/WER как в "
        "scripts/responses_api_analyze/metrics.py (агрегат — склейка нормализованных фрагментов "
        "и повторный nfkc_ws на склейке). Гипотезы: MinerU — output/mineru/*.md, Paddle — "
        "output/paddle/*.txt, GOT — output/got/*.txt (или output/got_benchmark/hypotheses/got/). "
        "Время — elapsed_sec в mineru_runs.jsonl, paddle_runs.jsonl, got_runs.jsonl. "
        "Столбец «Лучший CER» — минимум CER среди доступных движков на файле. "
        "CER по jiwer может быть >1."
    )

    odt = repo / "ocr-analyze.odt"
    buf = odt.read_bytes()
    with zipfile.ZipFile(io.BytesIO(buf), "r") as zin:
        content = zin.read("content.xml").decode("utf-8")

    insert = (
        f'<text:p text:style-name="P1">{xml_text(note)}</text:p>'
        f'<text:p text:style-name="P1"/>'
        f"{table_summary}"
        f'<text:p text:style-name="P1"/>'
        f'<text:p text:style-name="P2">Статистика по каждому изображению</text:p>'
        f'<text:p text:style-name="P1"/>'
        f"{table_per}"
        f'<text:p text:style-name="P1"/>'
    )

    block = find_report_block(content)
    if block is None:
        print("fill_ocr_analyze_odt: не найден блок отчёта (методика или ТаблицаСводка)", file=sys.stderr)
        return 1
    start, end = block
    tail = content[end:]
    new_content = content[:start] + insert + tail

    out_buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(buf), "r") as zin, zipfile.ZipFile(out_buf, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "content.xml":
                data = new_content.encode("utf-8")
            zi = zipfile.ZipInfo(item.filename, item.date_time)
            zi.compress_type = item.compress_type
            zout.writestr(zi, data)
    odt.write_bytes(out_buf.getvalue())
    print("Обновлено:", odt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
