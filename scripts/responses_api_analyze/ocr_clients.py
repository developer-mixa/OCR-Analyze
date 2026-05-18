"""Облачный OCR: AWS Textract, Google Vision, Aspose.OCR Cloud, FreeOCR.AI (RapidAPI)."""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Literal

_aspose_token: dict[str, Any] = {"access_token": None, "exp_at": 0.0}


def _guess_image_mime(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suf == ".webp":
        return "image/webp"
    if suf == ".gif":
        return "image/gif"
    if suf in (".tif", ".tiff"):
        return "image/tiff"
    return "image/png"


def ocr_textract_detect_document_text(*, image_path: Path, timeout: float = 60.0) -> tuple[str, dict[str, Any] | None, float]:
    """AWS Textract DetectDocumentText (байты изображения; многостраничный PDF — отдельный асинхронный поток в AWS)."""
    from botocore.config import Config

    import boto3

    t0 = time.perf_counter()
    region = os.environ.get("AWS_REGION", "us-east-1")
    cfg = Config(read_timeout=int(timeout), connect_timeout=10)
    client = boto3.client("textract", region_name=region, config=cfg)
    raw = image_path.read_bytes()
    resp = client.detect_document_text(Document={"Bytes": raw})
    elapsed = time.perf_counter() - t0
    lines: list[tuple[float, float, str]] = []
    for b in resp.get("Blocks") or []:
        if b.get("BlockType") != "LINE":
            continue
        text = (b.get("Text") or "").strip()
        if not text:
            continue
        geom = b.get("Geometry") or {}
        bb = geom.get("BoundingBox") or {}
        top = float(bb.get("Top", 0))
        left = float(bb.get("Left", 0))
        lines.append((top, left, text))
    lines.sort(key=lambda x: (x[0], x[1]))
    out = "\n".join(t for _, _, t in lines)
    meta: dict[str, Any] = {"transport": "aws_textract", "region": region, "api": "DetectDocumentText"}
    return out.strip(), meta, elapsed


def _google_vision_oauth_scopes() -> list[str]:
    raw = os.environ.get("GOOGLE_OAUTH_SCOPES", "https://www.googleapis.com/auth/cloud-vision").strip()
    return [s.strip() for s in raw.split(",") if s.strip()]


def _google_oauth_run_local_server(flow: Any) -> Any:
    """
    Фиксированный localhost-порт — его нужно добавить в Google Console как Authorized redirect URI
    (для типа «Веб-приложение»). Случайный port=0 даёт redirect_uri_mismatch.
    """
    base_port = int(os.environ.get("GOOGLE_OAUTH_LOCAL_SERVER_PORT", "8085"))
    host = (os.environ.get("GOOGLE_OAUTH_LOCAL_SERVER_HOST") or "localhost").strip() or "localhost"
    last_err: OSError | None = None
    for delta in range(20):
        port = base_port + delta
        try:
            return flow.run_local_server(
                host=host,
                port=port,
                open_browser=True,
                redirect_uri_trailing_slash=True,
            )
        except OSError as e:
            last_err = e
            continue
    raise OSError(f"Не удалось занять порт для OAuth (с {base_port}): {last_err}") from last_err


def _google_vision_oauth_credentials() -> Any:
    """OAuth 2.0 installed app: client_secret JSON + файл токена (refresh)."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    secret = Path(os.environ["GOOGLE_OAUTH_CLIENT_SECRETS_JSON"]).expanduser().resolve()
    if not secret.is_file():
        raise FileNotFoundError(f"GOOGLE_OAUTH_CLIENT_SECRETS_JSON: нет файла {secret}")
    token_path = Path(
        os.environ.get("GOOGLE_OAUTH_TOKEN_JSON") or (secret.parent / "google_vision_oauth_token.json")
    ).expanduser().resolve()
    scopes = _google_vision_oauth_scopes()

    creds: Any = None
    if token_path.is_file():
        creds = Credentials.from_authorized_user_file(str(token_path), scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(secret), scopes)
            creds = _google_oauth_run_local_server(flow)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")
    return creds


def ocr_google_vision_document(*, image_path: Path, timeout: float = 60.0) -> tuple[str, dict[str, Any] | None, float]:
    """Google Cloud Vision images:annotate, DOCUMENT_TEXT_DETECTION (API key или OAuth)."""
    import requests

    b64 = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
    body = {
        "requests": [
            {
                "image": {"content": b64},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION", "maxResults": 50}],
            }
        ]
    }
    t0 = time.perf_counter()
    if os.environ.get("GOOGLE_VISION_API_KEY"):
        url = f"https://vision.googleapis.com/v1/images:annotate?key={os.environ['GOOGLE_VISION_API_KEY']}"
        headers: dict[str, str] = {}
        auth_mode = "api_key"
    elif os.environ.get("GOOGLE_OAUTH_CLIENT_SECRETS_JSON"):
        creds = _google_vision_oauth_credentials()
        url = "https://vision.googleapis.com/v1/images:annotate"
        headers = {"Authorization": f"Bearer {creds.token}"}
        auth_mode = "oauth"
    else:
        raise RuntimeError("Задайте GOOGLE_VISION_API_KEY или GOOGLE_OAUTH_CLIENT_SECRETS_JSON")

    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    payload = r.json()
    responses = payload.get("responses") or []
    if not responses:
        raise RuntimeError(f"Vision API: пустой responses: {payload!s}"[:2000])
    err = responses[0].get("error")
    if err:
        raise RuntimeError(err.get("message") or str(err))
    ann = responses[0].get("fullTextAnnotation") or {}
    text = (ann.get("text") or "").strip()
    if not text and responses[0].get("textAnnotations"):
        text = (responses[0]["textAnnotations"][0].get("description") or "").strip()
    meta: dict[str, Any] = {
        "transport": "google_vision",
        "feature": "DOCUMENT_TEXT_DETECTION",
        "auth": auth_mode,
    }
    return text, meta, elapsed


def _aspose_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _aspose_bearer_token() -> str:
    cid = os.environ["ASPOSE_OCR_CLIENT_ID"]
    sec = os.environ["ASPOSE_OCR_CLIENT_SECRET"]
    now = time.time()
    if _aspose_token["access_token"] and now < float(_aspose_token["exp_at"]) - 30:
        return str(_aspose_token["access_token"])
    import requests

    r = requests.post(
        "https://api.aspose.cloud/connect/token",
        data={
            "grant_type": "client_credentials",
            "client_id": cid,
            "client_secret": sec,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    tok = data.get("access_token")
    if not tok:
        raise RuntimeError(f"Aspose token: {data!s}"[:500])
    _aspose_token["access_token"] = tok
    _aspose_token["exp_at"] = now + float(data.get("expires_in", 3600))
    return str(tok)


def ocr_aspose_cloud_image(*, image_path: Path, timeout: float = 120.0) -> tuple[str, dict[str, Any] | None, float]:
    """
    Aspose.OCR Cloud v5: POST RecognizeImage → GET по id (очередь).
    Триал: ASPOSE_OCR_USE_TRIAL=1 — RecognizeImageTrial (без ключа; часть слов маскируется).
    """
    import requests

    trial = _aspose_truthy("ASPOSE_OCR_USE_TRIAL")
    b64img = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
    settings = {
        "language": os.environ.get("ASPOSE_OCR_LANGUAGE", "Russian"),
        "makeSkewCorrect": os.environ.get("ASPOSE_OCR_DESKEW", "true").lower() == "true",
        "rotate": int(os.environ.get("ASPOSE_OCR_ROTATE", "0") or 0),
        "makeBinarization": os.environ.get("ASPOSE_OCR_BINARIZE", "false").lower() == "true",
        "makeContrastCorrection": os.environ.get("ASPOSE_OCR_CONTRAST", "true").lower() == "true",
        "makeUpsampling": os.environ.get("ASPOSE_OCR_UPSAMPLE", "false").lower() == "true",
        "makeSpellCheck": os.environ.get("ASPOSE_OCR_SPELL", "false").lower() == "true",
        "dsrMode": os.environ.get("ASPOSE_OCR_DSR_MODE", "DsrNoFilter"),
        "dsrConfidence": os.environ.get("ASPOSE_OCR_DSR_CONFIDENCE", "Default"),
        "resultType": os.environ.get("ASPOSE_OCR_RESULT_TYPE", "Text"),
    }
    post_url = (
        "https://api.aspose.cloud/v5.0/ocr/RecognizeImageTrial"
        if trial
        else "https://api.aspose.cloud/v5.0/ocr/RecognizeImage"
    )
    headers: dict[str, str] = {"Content-Type": "application/json", "Accept": "text/plain"}
    if not trial:
        headers["Authorization"] = f"Bearer {_aspose_bearer_token()}"
    t0 = time.perf_counter()
    pr = requests.post(
        post_url,
        headers=headers,
        json={"image": b64img, "settings": settings},
        timeout=timeout,
    )
    pr.raise_for_status()
    ct = (pr.headers.get("Content-Type") or "").lower()
    if "json" in ct:
        try:
            jd = pr.json()
            task_id = str(jd.get("id") or jd.get("Id") or "").strip()
        except Exception:
            task_id = ""
        if not task_id:
            task_id = (pr.text or "").strip().strip('"')
    else:
        task_id = (pr.text or "").strip().strip('"')
    if not task_id:
        raise RuntimeError(f"Aspose: пустой id задачи: {pr.status_code} {pr.text!s}"[:500])
    get_url = (
        f"https://api.aspose.cloud/v5.0/ocr/RecognizeImageTrial?id={task_id}"
        if trial
        else f"https://api.aspose.cloud/v5.0/ocr/RecognizeImage?id={task_id}"
    )
    get_headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if not trial:
        get_headers["Authorization"] = f"Bearer {_aspose_token['access_token']}"
    deadline = time.perf_counter() + float(os.environ.get("ASPOSE_OCR_POLL_MAX_SEC", "90"))
    poll = float(os.environ.get("ASPOSE_OCR_POLL_INTERVAL_SEC", "1.5"))
    data: dict[str, Any] | None = None
    while time.perf_counter() < deadline:
        gr = requests.get(get_url, headers=get_headers, timeout=timeout)
        gr.raise_for_status()
        data = gr.json()
        status = (data.get("taskStatus") or "").strip()
        if status == "Completed":
            break
        if status == "Error":
            raise RuntimeError(f"Aspose OCR error: {data.get('error') or data!s}"[:2000])
        time.sleep(poll)
    elapsed = time.perf_counter() - t0
    if not data or (data.get("taskStatus") or "").strip() != "Completed":
        raise RuntimeError(f"Aspose: таймаут ожидания результата id={task_id}")
    parts: list[str] = []
    for item in data.get("results") or []:
        if not isinstance(item, dict):
            continue
        raw_b64 = item.get("data")
        if isinstance(raw_b64, str) and raw_b64.strip():
            try:
                parts.append(base64.standard_b64decode(raw_b64).decode("utf-8", errors="replace"))
            except Exception:
                parts.append(raw_b64)
    text = "\n".join(p.strip() for p in parts if p).strip()
    meta: dict[str, Any] = {
        "transport": "aspose_ocr_cloud",
        "trial": trial,
        "task_id": task_id,
    }
    return text, meta, elapsed


def ocr_freeocr_rapidapi(
    *,
    image_path: Path,
    mode: Literal["ocr", "table"],
    timeout: float = 120.0,
) -> tuple[str, dict[str, Any] | None, float]:
    """FreeOCR.AI на RapidAPI: POST /ocr или /table.

    Язык в публичном API не задаётся (русский и др. — автоопределение на стороне сервиса).
    См. https://freeocr.ai/api — поля ``image`` и опционально ``format``.
    """
    import requests

    key = os.environ["FREEOCR_RAPIDAPI_KEY"]
    host = os.environ.get("FREEOCR_RAPIDAPI_HOST", "apis-freeocr-ai.p.rapidapi.com").strip()
    base = os.environ.get("FREEOCR_RAPIDAPI_BASE", f"https://{host}").rstrip("/")
    path = "/ocr" if mode == "ocr" else "/table"
    url = f"{base}{path}"
    t0 = time.perf_counter()
    mime = _guess_image_mime(image_path)
    headers = {"X-RapidAPI-Key": key, "X-RapidAPI-Host": host}
    form_data: dict[str, str] = {}
    ocr_fmt_default = "text"
    if mode == "ocr":
        fmt = os.environ.get("FREEOCR_OCR_FORMAT", ocr_fmt_default).strip().lower()
        if fmt not in ("json", "text", "docx"):
            fmt = ocr_fmt_default
        form_data["format"] = fmt
    else:
        fmt = os.environ.get("FREEOCR_TABLE_FORMAT", "json").strip().lower()
        if fmt not in ("json", "csv", "xlsx"):
            fmt = "json"
        form_data["format"] = fmt
    with image_path.open("rb") as fp:
        files = {"image": (image_path.name, fp, mime)}
        r = requests.post(url, headers=headers, files=files, data=form_data, timeout=timeout)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").lower()
    out_fmt = form_data.get("format", "")
    if mode == "ocr" and out_fmt == "docx":
        raise RuntimeError(
            "FreeOCR: format=docx возвращает бинарный DOCX; для бенчмарка задайте FREEOCR_OCR_FORMAT=json или text"
        )
    if mode == "table" and out_fmt == "xlsx":
        raise RuntimeError(
            "FreeOCR: format=xlsx возвращает бинарный XLSX; для скрипта задайте FREEOCR_TABLE_FORMAT=json или csv"
        )
    if mode == "table" and out_fmt == "csv":
        text = r.content.decode("utf-8", errors="replace").strip()
    elif "json" in ctype:
        body = r.json()
        if mode == "ocr":
            if isinstance(body, str):
                text = body.strip()
            else:
                text = (body.get("text") if isinstance(body, dict) else None) or json.dumps(
                    body, ensure_ascii=False
                )
                if isinstance(text, str):
                    text = text.strip()
        else:
            text = json.dumps(body, ensure_ascii=False, indent=2) if body is not None else ""
    elif ctype.startswith("text/"):
        text = r.text.strip()
    else:
        text = r.text.strip()
    meta: dict[str, Any] = {
        "transport": "freeocr_rapidapi",
        "mode": mode,
        "host": host,
        "format": form_data.get("format"),
        "language": "auto",
    }
    return text, meta, elapsed
