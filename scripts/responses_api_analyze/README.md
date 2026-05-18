# Облачный OCR-бенчмарк (`responses_api_analyze`)

Скрипт `run_ocr_api_benchmark.py` вызывает **только внешние SaaS API** (без локального vLLM и без своего OpenAI-compatible сервера). Имя каталога историческое.

## Провайдеры

| CLI | Сервис | Переменные окружения |
|-----|--------|----------------------|
| **textract** | Amazon Textract `DetectDocumentText` (байты файла) | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, опционально `AWS_REGION` (по умолчанию `us-east-1`) |
| **google_vision** | Google Cloud Vision `images:annotate`, `DOCUMENT_TEXT_DETECTION` | **`GOOGLE_VISION_API_KEY`** **или** OAuth: **`GOOGLE_OAUTH_CLIENT_SECRETS_JSON`** (путь к `client_secret_….apps.googleusercontent.com.json`). При первом запуске откроется браузер; токен сохранится в `GOOGLE_OAUTH_TOKEN_JSON` или рядом с JSON: `google_vision_oauth_token.json`. Опционально: `GOOGLE_OAUTH_SCOPES` (по умолчанию `https://www.googleapis.com/auth/cloud-vision`). Если заданы и ключ, и OAuth, используется **API key**. |
| **aspose_ocr** | [Aspose.OCR Cloud](https://docs.aspose.cloud/ocr/recognize-image/) v5 `RecognizeImage` + опрос результата | `ASPOSE_OCR_CLIENT_ID`, `ASPOSE_OCR_CLIENT_SECRET` **или** триал `ASPOSE_OCR_USE_TRIAL=1` (без ключей; часть текста маскируется). `ASPOSE_OCR_LANGUAGE` по умолчанию **`Russian`** ([список языков](https://docs.aspose.cloud/ocr/supported-languages/)); для англ. текста — `English`. Ещё: `ASPOSE_OCR_RESULT_TYPE`, `ASPOSE_OCR_DSR_MODE` (по умолчанию `DsrNoFilter`; [DSR](https://docs.aspose.cloud/ocr/structure-analysis/)), `ASPOSE_OCR_POLL_MAX_SEC`, … |
| **freeocr** | [FreeOCR.AI](https://freeocr.ai/api) на RapidAPI `POST /ocr` | `FREEOCR_RAPIDAPI_KEY`; опционально `FREEOCR_RAPIDAPI_HOST`, `FREEOCR_RAPIDAPI_BASE`. **Язык в API не передаётся** — сервис сам определяет язык (в т.ч. русский). `FREEOCR_OCR_FORMAT` по умолчанию **`text`** (plain text без Markdown; см. документацию API); иначе `json` или `docx` (docx в этом скрипте не поддержан — будет ошибка). |
| **freeocr_table** | Тот же API, `POST /table` (таблицы → JSON в файле гипотезы) | те же, что у **freeocr**. `FREEOCR_TABLE_FORMAT` по умолчанию **`json`**; `csv` — текст в гипотезе; `xlsx` в скрипте не поддержан. |

Псевдонимы: `amazon_textract` → `textract`, `ocr_table_api` → `freeocr_table`.

`API_OCR_USE_ALL=1` включает все **пять** провайдеров; без полного набора ключей `check_env` завершит процесс с ошибкой — задайте только нужные и `API_OCR_PROVIDERS=textract,google_vision`.

### Google Vision OAuth и `redirect_uri_mismatch`

Скрипт поднимает локальный callback на **фиксированном порту** (по умолчанию **8085**, см. `GOOGLE_OAUTH_LOCAL_SERVER_PORT`). В Google Cloud Console → OAuth 2.0 Client → **Authorized redirect URIs** для типа **Веб-приложение** добавьте **`http://localhost:8085/`** (и при смене порта — URI с тем же портом). Раньше использовался случайный порт (`localhost:44343`), из‑за этого и возникал mismatch.

Надёжнее для CLI создать клиент типа **Desktop** и скачать JSON с секцией `installed`.

## Ограничения по форматам

- Скрипт ориентирован на **файлы из каталога** (`--glob`, по умолчанию `*.png`). **Многостраничный PDF в Textract** через этот скрипт **не** реализован (нужны S3 + асинхронные job в AWS). **PDF в Vision** — отдельный batch/async API; здесь только **изображения** в одном запросе.
- Для полноценного PDF-пайплайна используйте конвертацию в изображения по страницам или расширение скрипта отдельно.

## Запуск

```bash
pip install boto3 requests jiwer pillow google-auth google-auth-oauthlib

export GOOGLE_VISION_API_KEY=...
python scripts/responses_api_analyze/run_ocr_api_benchmark.py \
  --providers google_vision \
  --input-dir input/data/1 \
  --output-dir output/api_ocr_benchmark
```

Только env: см. `.env.example` в корне — файл **`.env` в корне репозитория** скрипт подхватывает сам при запуске (переменные, уже заданные в shell, не перезаписываются). Флаги CLI переопределяют env.

## Артефакты

`api_ocr_runs.jsonl`, `hypotheses/<provider>/`, `api_ocr_summaries.json`; эталоны рядом с изображениями: `stem.ref.txt` / `stem.ref.md` / `stem.txt` / `stem.md` (приоритет слева направо).
