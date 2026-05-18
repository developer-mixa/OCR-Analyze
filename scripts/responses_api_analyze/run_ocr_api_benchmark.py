#!/usr/bin/env python3
"""
Прогон OCR по **облачным** API (без локальных vLLM/серверов):

- Amazon Textract (`DetectDocumentText`, изображение как байты)
- Google Cloud Vision (`DOCUMENT_TEXT_DETECTION`)
- Aspose.OCR Cloud (v5 RecognizeImage + опрос результата; опционально триал)
- FreeOCR.AI через RapidAPI: текст (`/ocr`) и таблицы (`/table`)

Метрики — как в model_analyze/task03 (CER, Accuracy, Final Score, …), плюс JSONL по каждой паре (файл×провайдер).

Переменные из файла **`.env` в корне репозитория** подхватываются автоматически при запуске (уже заданные в shell не перезаписываются). Иначе: `set -a && source .env && set +a`.

Зависимости:
  pip install boto3 requests jiwer pillow google-auth google-auth-oauthlib

Полный набор: export API_OCR_USE_ALL=1 (нужны все ключи из README / .env.example).

Примеры:
  export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-east-1
  python scripts/responses_api_analyze/run_ocr_api_benchmark.py --providers textract --input-dir input/data/1

  export GOOGLE_VISION_API_KEY=...
  python scripts/responses_api_analyze/run_ocr_api_benchmark.py --providers google_vision --input-dir input/data/1

  # Vision через OAuth (client_secret*.json из консоли Google):
  export GOOGLE_OAUTH_CLIENT_SECRETS_JSON=client_secret_....apps.googleusercontent.com.json
  python scripts/responses_api_analyze/run_ocr_api_benchmark.py --providers google_vision --input-dir input/data/1

  export ASPOSE_OCR_CLIENT_ID=... ASPOSE_OCR_CLIENT_SECRET=...
  python scripts/responses_api_analyze/run_ocr_api_benchmark.py --providers aspose_ocr --input-dir input/data/1

  export FREEOCR_RAPIDAPI_KEY=...
  # опционально: FREEOCR_OCR_FORMAT=json|text (по умолчанию в коде text)
  python scripts/responses_api_analyze/run_ocr_api_benchmark.py --providers freeocr,freeocr_table --input-dir input/data/1

"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from metrics import build_metric_row, char_error_rate, normalize_text
from ocr_clients import (
    ocr_aspose_cloud_image,
    ocr_freeocr_rapidapi,
    ocr_google_vision_document,
    ocr_textract_detect_document_text,
)

PROVIDER_ALIASES = {
    "textract": "textract",
    "amazon_textract": "textract",
    "aws_textract": "textract",
    "google_vision": "google_vision",
    "google": "google_vision",
    "vision": "google_vision",
    "gcp_vision": "google_vision",
    "aspose_ocr": "aspose_ocr",
    "aspose": "aspose_ocr",
    "freeocr": "freeocr",
    "freeocr_ai": "freeocr",
    "freeocr_rapid": "freeocr",
    "freeocr_table": "freeocr_table",
    "ocr_table_api": "freeocr_table",
    "table_api": "freeocr_table",
}

ALL_CANONICAL_PROVIDERS: tuple[str, ...] = (
    "textract",
    "google_vision",
    "aspose_ocr",
    "freeocr",
    "freeocr_table",
)


def load_repo_dotenv() -> None:
    """Читает корневой .env репозитория; не перезаписывает уже установленные переменные окружения."""
    root = Path(__file__).resolve().parents[2]
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


def env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def load_reference(img: Path) -> str | None:
    stem = img.stem
    for name in (f"{stem}.ref.txt", f"{stem}.ref.md", f"{stem}.txt", f"{stem}.md"):
        p = img.parent / name
        if p.is_file():
            t = p.read_text(encoding="utf-8")
            if t.strip():
                return t
    return None


def run_provider(canonical: str, image: Path) -> tuple[str, dict | None, float, str | None]:
    """Возвращает (text, meta, elapsed, error)."""
    try:
        if canonical == "textract":
            text, meta, sec = ocr_textract_detect_document_text(image_path=image)
            return text, meta, sec, None
        if canonical == "google_vision":
            text, meta, sec = ocr_google_vision_document(image_path=image)
            return text, meta, sec, None
        if canonical == "aspose_ocr":
            text, meta, sec = ocr_aspose_cloud_image(image_path=image)
            return text, meta, sec, None
        if canonical == "freeocr":
            text, meta, sec = ocr_freeocr_rapidapi(image_path=image, mode="ocr")
            return text, meta, sec, None
        if canonical == "freeocr_table":
            text, meta, sec = ocr_freeocr_rapidapi(image_path=image, mode="table")
            return text, meta, sec, None
    except Exception as e:
        return "", None, 0.0, repr(e)
    return "", None, 0.0, f"unknown provider {canonical!r}"


def normalize_provider_list(raw: list[str]) -> list[str]:
    out: list[str] = []
    for p in raw:
        key = p.strip().lower().replace(" ", "")
        if key not in PROVIDER_ALIASES:
            raise SystemExit(f"Неизвестный провайдер {p!r}. Допустимые ключи: {sorted(set(PROVIDER_ALIASES))}")
        c = PROVIDER_ALIASES[key]
        if c not in out:
            out.append(c)
    return out


def check_env(providers: list[str]) -> None:
    need: dict[str, list[str]] = {
        "textract": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
        "google_vision": [],
        "freeocr": ["FREEOCR_RAPIDAPI_KEY"],
        "freeocr_table": ["FREEOCR_RAPIDAPI_KEY"],
        "aspose_ocr": [],
    }
    for p in providers:
        for k in need.get(p, []):
            if not os.environ.get(k):
                raise SystemExit(f"Провайдер {p!r} требует переменную окружения {k}")
    if "google_vision" in providers:
        if os.environ.get("GOOGLE_VISION_API_KEY"):
            pass
        elif os.environ.get("GOOGLE_OAUTH_CLIENT_SECRETS_JSON"):
            p = Path(os.environ["GOOGLE_OAUTH_CLIENT_SECRETS_JSON"]).expanduser()
            if not p.is_file():
                raise SystemExit(f"google_vision: нет файла GOOGLE_OAUTH_CLIENT_SECRETS_JSON={p}")
        else:
            raise SystemExit(
                "google_vision: задайте GOOGLE_VISION_API_KEY или GOOGLE_OAUTH_CLIENT_SECRETS_JSON (путь к client_secret JSON)"
            )
    if "aspose_ocr" in providers:
        if env_truthy("ASPOSE_OCR_USE_TRIAL"):
            pass
        elif not os.environ.get("ASPOSE_OCR_CLIENT_ID") or not os.environ.get("ASPOSE_OCR_CLIENT_SECRET"):
            raise SystemExit(
                "aspose_ocr: задайте ASPOSE_OCR_CLIENT_ID и ASPOSE_OCR_CLIENT_SECRET "
                "или включите триал ASPOSE_OCR_USE_TRIAL=1"
            )


def _summary_label(prov: str) -> str:
    if prov == "textract":
        return f"aws_textract:{os.environ.get('AWS_REGION', 'us-east-1')}"
    if prov == "google_vision":
        if os.environ.get("GOOGLE_OAUTH_CLIENT_SECRETS_JSON"):
            return "google_vision:DOCUMENT_TEXT_DETECTION(oauth)"
        return "google_vision:DOCUMENT_TEXT_DETECTION(api_key)"
    if prov == "aspose_ocr":
        return "aspose_ocr_cloud:v5" + ("_trial" if env_truthy("ASPOSE_OCR_USE_TRIAL") else "")
    if prov == "freeocr":
        fo = (os.environ.get("FREEOCR_OCR_FORMAT") or "text").strip().lower()
        if fo not in ("json", "text", "docx"):
            fo = "text"
        return f"freeocr_rapidapi:/ocr(format={fo},language=auto)"
    if prov == "freeocr_table":
        ft = (os.environ.get("FREEOCR_TABLE_FORMAT") or "json").strip().lower()
        if ft not in ("json", "csv", "xlsx"):
            ft = "json"
        return f"freeocr_rapidapi:/table(format={ft})"
    return f"api:{prov}"


def main() -> int:
    load_repo_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Каталог с изображениями. Иначе $API_OCR_INPUT_DIR, иначе input/data/1",
    )
    ap.add_argument(
        "--glob",
        default=None,
        help="Шаблон файлов. Иначе $API_OCR_GLOB, иначе *.png",
    )
    ap.add_argument(
        "--providers",
        default=None,
        help=f"Через запятую: {','.join(ALL_CANONICAL_PROVIDERS)}. Иначе $API_OCR_PROVIDERS; или $API_OCR_USE_ALL=1",
    )
    ap.add_argument(
        "--normalize",
        default=None,
        help="Нормализация для CER. Иначе $API_OCR_NORMALIZE, иначе nfkc_ws",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Куда писать артефакты. Иначе $API_OCR_OUTPUT_DIR, иначе output/api_ocr_benchmark",
    )
    ap.add_argument(
        "--internal-parser",
        default=None,
        help="Поле «Внутренний парсер» в сводке. Иначе $API_OCR_INTERNAL_PARSER, иначе none",
    )
    args = ap.parse_args()

    providers_raw = args.providers
    if providers_raw is None:
        providers_raw = os.environ.get("API_OCR_PROVIDERS")
    if not (providers_raw and str(providers_raw).strip()):
        if env_truthy("API_OCR_USE_ALL"):
            providers = list(ALL_CANONICAL_PROVIDERS)
        else:
            plist = ",".join(ALL_CANONICAL_PROVIDERS)
            raise SystemExit(
                f"Задайте провайдеры: --providers …, или export API_OCR_PROVIDERS=…, "
                f"или export API_OCR_USE_ALL=1 (все: {plist}; задайте ключи в env)."
            )
    else:
        providers = normalize_provider_list([x for x in str(providers_raw).split(",") if x.strip()])
    check_env(providers)

    inp = args.input_dir
    if inp is None:
        d = os.environ.get("API_OCR_INPUT_DIR", "").strip()
        inp = Path(d) if d else Path("input/data/1")
    if not inp.is_dir():
        raise SystemExit(f"Нет каталога: {inp}")

    out_root = args.output_dir
    if out_root is None:
        d = os.environ.get("API_OCR_OUTPUT_DIR", "").strip()
        out_root = Path(d) if d else Path("output/api_ocr_benchmark")
    hyp_root = out_root / "hypotheses"
    hyp_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "api_ocr_runs.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    glob_pat = args.glob
    if glob_pat is None:
        g = os.environ.get("API_OCR_GLOB", "").strip()
        glob_pat = g if g else "*.png"

    normalize = args.normalize
    if normalize is None:
        n = os.environ.get("API_OCR_NORMALIZE", "").strip()
        normalize = n if n else "nfkc_ws"

    internal_parser = args.internal_parser
    if internal_parser is None:
        ip = os.environ.get("API_OCR_INTERNAL_PARSER", "").strip()
        internal_parser = ip if ip else "none"

    images = sorted(inp.glob(glob_pat))
    if not images:
        print("Нет изображений по шаблону", glob_pat, "в", inp)
        return 0

    corpus: dict[str, tuple[list[str], list[str]]] = {p: ([], []) for p in providers}

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for img in images:
            ref_raw = load_reference(img)
            had_ref = bool(ref_raw and ref_raw.strip())
            ref_n = normalize_text(ref_raw or "", normalize)
            for prov in providers:
                text, meta, sec, err = run_provider(prov, img)
                hdir = hyp_root / prov
                hdir.mkdir(parents=True, exist_ok=True)
                hpath = hdir / f"{img.stem}.md"
                if err is None:
                    hpath.write_text(text, encoding="utf-8")
                row = {
                    "file": img.name,
                    "provider": prov,
                    "error": err,
                    "elapsed_sec": round(sec, 4) if err is None else None,
                    "hypothesis_path": str(hpath) if err is None else None,
                    "meta": meta,
                    "had_reference": had_ref,
                    "CER": None,
                    "char_accuracy": None,
                }
                if err is None and had_ref:
                    cer = char_error_rate(ref_n, normalize_text(text, normalize))
                    row["CER"] = round(cer, 6)
                    row["char_accuracy"] = round(max(0.0, min(1.0, 1.0 - cer)), 6)
                    corpus[prov][0].append(ref_n)
                    corpus[prov][1].append(normalize_text(text, normalize))
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")

    summaries: dict[str, dict] = {}
    for prov in providers:
        refs, hyps = corpus[prov]
        label = _summary_label(prov)
        comment = f"providers_script=api_benchmark; n_files={len(images)}; n_with_cer={len(refs)}"
        if not refs:
            summaries[prov] = {
                "Model": label,
                "Final Score": None,
                "Accuracy": None,
                "CER": None,
                "Unit Test Rate": None,
                "Внутренний парсер": internal_parser,
                "Токены": {"note": "нет эталонов — CER не считался"},
                "Доп. Комментарии": comment,
                "Скорость": None,
            }
            continue
        hyp_concat = "\n".join(hyps)
        row = build_metric_row(
            ref_raw="\n".join(refs),
            hyp_raw=hyp_concat,
            normalize=normalize,
            model=label,
            internal_parser=internal_parser,
            extra_comment=comment,
        )
        row["Скорость"] = {
            "note": "среднее по файлам см. в JSONL elapsed_sec",
        }
        summaries[prov] = row

    (out_root / "api_ocr_summaries.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("JSONL:", jsonl_path.resolve())
    print("Summaries:", (out_root / "api_ocr_summaries.json").resolve())
    print("Hypotheses:", hyp_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
