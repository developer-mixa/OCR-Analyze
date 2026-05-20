#!/usr/bin/env python3
"""
Прогон **Yandex Vision OCR** (REST ``recognizeText``) по изображениям и те же метрики, что у
``paddle_image_benchmark`` / ``mineru_image_benchmark``.

Пишет плоский текст в ``hypotheses/yandex_vision/<stem>.txt``, JSONL ``yandex_vision_runs.jsonl``,
сводку ``yandex_vision_summaries.json`` (ключ **``yandex_vision``**), плюс
``yandex_vision_hypotheses_raw.json`` / ``yandex_vision_hypotheses_concat.txt``.

**Эталоны:** ``<stem>.ref.txt`` → ``<stem>.ref.md`` → ``<stem>.txt`` → ``<stem>.md``.

**Аутентификация** (один из вариантов):

- IAM: ``YANDEX_OCR_IAM_TOKEN`` или ``YC_IAM_TOKEN`` — заголовок ``Authorization: Bearer …``;
  каталог: ``YANDEX_OCR_FOLDER_ID`` или ``YANDEX_GPT_FOLDER_ID`` в ``x-folder-id``.
- API-ключ: ``YANDEX_OCR_API_KEY`` или ``YANDEX_GPT_API_KEY`` — ``Authorization: Api-Key …``;
  каталог — те же переменные (для ключа пользователя обычно нужен ``x-folder-id``).

**Документация:** https://aistudio.yandex.ru/docs/ru/vision/quickstart.html

**Зависимости:** ``pip install jiwer`` (HTTP — ``urllib`` из стандартной библиотеки).

Пример из корня репозитория::

  export YANDEX_GPT_FOLDER_ID=b1g...
  export YANDEX_GPT_API_KEY=AQVN...
  python scripts/yandex_vision_image_benchmark.py \\
    --input-dir input/data/1 \\
    --output-dir output/yandex_vision \\
    --model page
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_RECOGNIZE_URL = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"


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


def mime_type_for_path(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in (".jpg", ".jpeg"):
        return "JPEG"
    if suf == ".png":
        return "PNG"
    if suf == ".pdf":
        return "PDF"
    return "JPEG"


def resolve_auth_headers(
    *,
    folder_id: str | None,
    iam_token: str | None,
    api_key: str | None,
    data_logging: bool,
) -> tuple[str | None, dict[str, str]]:
    """
    Возвращает (ошибка или None, headers без Content-Length — добавит urllib).
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if data_logging:
        headers["x-data-logging-enabled"] = "true"
    fid = (folder_id or "").strip()
    if iam_token and iam_token.strip():
        headers["Authorization"] = f"Bearer {iam_token.strip()}"
        if fid:
            headers["x-folder-id"] = fid
        return None, headers
    if api_key and api_key.strip():
        headers["Authorization"] = f"Api-Key {api_key.strip()}"
        if fid:
            headers["x-folder-id"] = fid
        return None, headers
    return (
        "Нет кредов: задайте YANDEX_OCR_IAM_TOKEN (или YC_IAM_TOKEN) либо "
        "YANDEX_OCR_API_KEY / YANDEX_GPT_API_KEY",
        {},
    )


def parse_language_codes(s: str) -> list[str]:
    s = (s or "").strip()
    if s == "*":
        return ["*"]
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    return parts or ["*"]


def _lines_from_text_annotation(ta: dict) -> str:
    lines_out: list[str] = []
    for block in ta.get("blocks") or []:
        for line in block.get("lines") or []:
            txt = (line.get("text") or "").strip()
            if txt:
                lines_out.append(txt)
    return "\n".join(lines_out)


def text_from_yandex_vision_response(obj: dict) -> str:
    """Извлекает плоский текст из JSON ответа recognizeText (синхронный)."""
    if not isinstance(obj, dict):
        return ""
    result = obj.get("result")
    if not isinstance(result, dict):
        result = obj

    pages = result.get("pages")
    if isinstance(pages, list) and pages:
        chunks: list[str] = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            ta = page.get("textAnnotation")
            if not isinstance(ta, dict):
                continue
            chunk = _text_from_single_annotation(ta)
            if chunk:
                chunks.append(chunk)
        return "\n\n".join(chunks)

    ta = result.get("textAnnotation")
    if isinstance(ta, dict):
        return _text_from_single_annotation(ta)
    return ""


def _text_from_single_annotation(ta: dict) -> str:
    ft = (ta.get("fullText") or "").strip()
    if ft:
        return ft
    md = ta.get("markdown")
    if md is not None and str(md).strip():
        return str(md).strip()
    return _lines_from_text_annotation(ta)


def post_recognize(
    *,
    url: str,
    headers: dict[str, str],
    image_bytes: bytes,
    mime_type: str,
    model: str,
    language_codes: list[str],
    timeout_sec: float,
) -> tuple[str | None, dict]:
    """Возвращает (ошибка или None, распарсенный JSON)."""
    payload = {
        "mimeType": mime_type,
        "languageCodes": language_codes,
        "model": model,
        "content": base64.b64encode(image_bytes).decode("ascii"),
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")[:4000]
        except OSError:
            detail = ""
        return f"HTTP {e.code}: {detail or e.reason}", {}
    except urllib.error.URLError as e:
        return f"URL error: {e}", {}
    except TimeoutError:
        return "timeout", {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return f"JSON decode: {e}; body[:500]={raw[:500]!r}", {}
    return None, data


def run_yandex_vision_on_file(
    *,
    image: Path,
    url: str,
    headers: dict[str, str],
    model: str,
    language_codes: list[str],
    timeout_sec: float,
) -> tuple[str | None, str]:
    try:
        blob = image.read_bytes()
    except OSError as e:
        return f"read image: {e}", ""
    mime = mime_type_for_path(image)
    err, data = post_recognize(
        url=url,
        headers=headers,
        image_bytes=blob,
        mime_type=mime,
        model=model,
        language_codes=language_codes,
        timeout_sec=timeout_sec,
    )
    if err:
        return err, ""
    # Ошибки API в теле
    if isinstance(data, dict) and data.get("error"):
        return json.dumps(data.get("error"), ensure_ascii=False), ""
    text = text_from_yandex_vision_response(data)
    if not text.strip():
        return "пустой текст OCR (проверьте model/mimeType/ответ в сыром JSON)", ""
    return None, text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=None, help="Каталог с изображениями (по умолчанию input/data/1)")
    p.add_argument("--glob", default="*.png", help='Шаблон файлов, напр. "*.png"')
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Куда писать артефакты (по умолчанию output/yandex_vision)",
    )
    p.add_argument(
        "--recognize-url",
        default=os.environ.get("YANDEX_OCR_RECOGNIZE_URL", DEFAULT_RECOGNIZE_URL),
        help="URL метода recognizeText",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("YANDEX_OCR_MODEL", "page"),
        help="Модель Vision OCR: page, table, markdown, handwritten, … (см. документацию)",
    )
    p.add_argument(
        "--language-codes",
        default=os.environ.get("YANDEX_OCR_LANGUAGE_CODES", "ru,en"),
        help='Языки: через запятую (ru,en) или "*" для авто',
    )
    p.add_argument(
        "--folder-id",
        default=None,
        help="Переопределить x-folder-id (иначе env YANDEX_OCR_FOLDER_ID / YANDEX_GPT_FOLDER_ID)",
    )
    p.add_argument(
        "--iam-token",
        default=None,
        help="Переопределить IAM (иначе YANDEX_OCR_IAM_TOKEN / YC_IAM_TOKEN)",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="Переопределить API-ключ (иначе YANDEX_OCR_API_KEY / YANDEX_GPT_API_KEY)",
    )
    p.add_argument(
        "--no-data-logging-header",
        action="store_true",
        help="Не отправлять x-data-logging-enabled",
    )
    p.add_argument(
        "--timeout-sec",
        type=float,
        default=float(os.environ.get("YANDEX_OCR_TIMEOUT_SEC", "120")),
        help="Таймаут HTTP на файл",
    )
    p.add_argument(
        "--normalize",
        default=os.environ.get("YANDEX_OCR_NORMALIZE", "nfkc_ws"),
        help="Режим нормализации для CER",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Не вызывать API; пересчитать метрики по hypotheses/yandex_vision/<stem>.txt",
    )
    return p.parse_args()


def main() -> int:
    load_repo_dotenv()
    _inject_metrics_path()
    from metrics import build_metric_row, char_error_rate, normalize_text

    args = parse_args()
    inp = args.input_dir or Path(os.environ.get("YANDEX_OCR_INPUT_DIR", "input/data/1"))
    if not inp.is_dir():
        raise SystemExit(f"Нет каталога: {inp.resolve()}")

    folder = args.folder_id or os.environ.get("YANDEX_OCR_FOLDER_ID") or os.environ.get("YANDEX_GPT_FOLDER_ID")
    iam = args.iam_token or os.environ.get("YANDEX_OCR_IAM_TOKEN") or os.environ.get("YC_IAM_TOKEN")
    api_key = args.api_key or os.environ.get("YANDEX_OCR_API_KEY") or os.environ.get("YANDEX_GPT_API_KEY")

    headers: dict[str, str] = {}
    if not args.dry_run:
        auth_err, headers = resolve_auth_headers(
            folder_id=folder,
            iam_token=iam,
            api_key=api_key,
            data_logging=not args.no_data_logging_header,
        )
        if auth_err:
            raise SystemExit(auth_err)

    lang_codes = parse_language_codes(args.language_codes)
    out_root = args.output_dir or Path(os.environ.get("YANDEX_VISION_OUTPUT_DIR", "output/yandex_vision"))
    out_root.mkdir(parents=True, exist_ok=True)
    hyp_root = out_root / "hypotheses" / "yandex_vision"
    hyp_root.mkdir(parents=True, exist_ok=True)

    images = sorted(inp.glob(args.glob))
    if not images:
        print("Нет файлов по шаблону", args.glob, "в", inp)
        return 0

    jsonl_path = out_root / "yandex_vision_runs.jsonl"
    refs_concat: list[str] = []
    hyps_concat: list[str] = []
    elapsed_ok: list[float] = []
    n_files_with_reference = 0

    model_label = f"yandex_vision:{args.model}:lang={args.language_codes}"
    raw_hypotheses: dict[str, str] = {}

    if args.dry_run:
        print("Режим --dry-run: API не вызывается. Нужны файлы", hyp_root / "<stem>.txt")

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
                err, text = run_yandex_vision_on_file(
                    image=img,
                    url=str(args.recognize_url).strip(),
                    headers=headers,
                    model=str(args.model).strip(),
                    language_codes=lang_codes,
                    timeout_sec=float(args.timeout_sec),
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
                "provider": "yandex_vision",
                "error": err,
                "elapsed_sec": elapsed_sec,
                "hypothesis_path": str(hyp_path) if err is None else None,
                "meta": {
                    "model": str(args.model).strip(),
                    "language_codes": lang_codes,
                    "recognize_url": str(args.recognize_url).strip(),
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
        f"yandex_vision_image_benchmark; n_files={len(images)}; n_with_cer={n_cer}; "
        f"model={args.model}; languages={lang_codes}"
    )
    if not refs_concat:
        if args.dry_run:
            token_note = (
                "режим --dry-run: нет hypotheses/yandex_vision/<stem>.txt — сначала полный прогон без --dry-run."
            )
        elif n_files_with_reference > 0:
            token_note = "эталоны есть, но нет успешных гипотез — см. error в yandex_vision_runs.jsonl."
        else:
            token_note = "нет эталонов рядом с изображениями — CER не считался"
        summary = {
            "Model": model_label,
            "Final Score": None,
            "Accuracy": None,
            "CER": None,
            "Unit Test Rate": None,
            "Внутренний парсер": "yandex_vision_rest",
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
            internal_parser="yandex_vision_rest",
            extra_comment=comment,
        )
        if elapsed_ok:
            summary["Скорость"] = {
                "mean_elapsed_sec_per_file": round(statistics.mean(elapsed_ok), 4),
                "total_elapsed_sec": round(sum(elapsed_ok), 3),
                "n_files_timed": len(elapsed_ok),
            }
        summary["_run_mode"] = {"dry_run": bool(args.dry_run)}

    bundle_json = out_root / "yandex_vision_hypotheses_raw.json"
    bundle_txt = out_root / "yandex_vision_hypotheses_concat.txt"
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
        "summaries_json": str((out_root / "yandex_vision_summaries.json").resolve()),
    }
    if raw_hypotheses:
        out_meta["hypotheses_bundle_json"] = str(bundle_json.resolve())
        out_meta["hypotheses_concat_txt"] = str(bundle_txt.resolve())
        out_meta["n_bundled_files"] = len(raw_hypotheses)
    summary["_outputs"] = out_meta

    (out_root / "yandex_vision_summaries.json").write_text(
        json.dumps({"yandex_vision": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Записано:", jsonl_path)
    print("Сводка:", out_root / "yandex_vision_summaries.json")
    print("Гипотезы (текст):", hyp_root)
    if raw_hypotheses:
        print("Сырой текст (JSON):", bundle_json.resolve())
        print("Сырой текст (склейка):", bundle_txt.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
