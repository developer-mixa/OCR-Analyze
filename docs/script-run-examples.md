# Примеры запуска скриптов

Команды ниже предполагают **текущий каталог = корень репозитория** (`AnalytiscNeuroOcr`).

- **Локальные / тяжёлые пайплайны** — каталог `scripts/model_analyze/` (короткие обёртки оставлены в `scripts/task01_*.py` и т.д. и вызывают файлы из `model_analyze/`).
- **Облачный OCR (AWS / Google / Aspose / FreeOCR RapidAPI)** — `scripts/responses_api_analyze/` (см. конец файла).

---

## task01 — Tesseract, метрики по изображениям

**Зависимости:** `pip install pillow pytesseract` + системный `tesseract` и языки (`rus`, `eng` и т.д.).

Базовый прогон по умолчанию (`input/data/1`, `*.png`), JSONL + сводка по-русски:

```bash
python scripts/model_analyze/task01_ocr_tesseract_baseline_metrics.py \
  --input-dir "input/data/1" \
  --output-jsonl output/task01_tesseract_metrics.jsonl \
  --print-summary \
  --print-summary-ru
```

Только JSON-сводка:

```bash
python scripts/model_analyze/task01_ocr_tesseract_baseline_metrics.py --input-dir "input/data/1" --print-summary
```

Другие языки и режимы Tesseract:

```bash
python scripts/model_analyze/task01_ocr_tesseract_baseline_metrics.py \
  --input-dir "input/data/1" \
  --lang "rus+eng" \
  --oem 3 \
  --psm 6 \
  --output-jsonl output/task01_tesseract_metrics.jsonl
```

Увеличить изображение перед OCR:

```bash
python scripts/model_analyze/task01_ocr_tesseract_baseline_metrics.py \
  --input-dir "input/data/1" \
  --scale 1.5 \
  --output-jsonl output/task01_tesseract_scale15.jsonl \
  --print-summary-ru
```

---

## task02 — Docling, те же поля метрик (другая семантика confidence)

**Зависимости:** `pip install docling pillow` (установка тяжёлая).

```bash
python scripts/model_analyze/task02_docling_baseline_metrics.py \
  --input-dir "input/data/1" \
  --glob "*.png" \
  --output-jsonl output/task02_docling_metrics.jsonl \
  --print-summary \
  --print-summary-ru
```

PDF (если положить файлы в каталог и сменить glob):

```bash
python scripts/model_analyze/task02_docling_baseline_metrics.py \
  --input-dir "input/data/pdf" \
  --glob "*.pdf" \
  --output-jsonl output/task02_docling_pdf.jsonl \
  --print-summary-ru
```

Метка пайплайна в поле `tesseract_config` (префикс `docling:`):

```bash
python scripts/model_analyze/task02_docling_baseline_metrics.py \
  --input-dir "input/data/1" \
  --pipeline-note "DocumentConverter(default)+manual" \
  --output-jsonl output/task02_docling_metrics.jsonl
```

---

## Черновики эталона — Tesseract + Docling (локально)

**Зависимости:** `pip install pillow pytesseract docling` + системный `tesseract` с языками.

Рядом с каждым PNG создаётся `stem.ref.draft.docling.md` (структура markdown) и `stem.ref.draft.tesseract.txt` (плоский текст). После правки сохраните эталон как `stem.ref.txt` или `stem.ref.md` (облачный бенчмарк `run_ocr_api_benchmark.py` ищет: `ref.txt` → `ref.md` → `stem.txt` → `stem.md`).

```bash
python scripts/draft_reference_from_local_ocr.py \
  --input-dir "input/data/1" \
  --print-summary
```

Один файл с двумя разделами:

```bash
python scripts/draft_reference_from_local_ocr.py --input-dir "input/data/1" --combined-md --force
```

Только Docling или только Tesseract:

```bash
python scripts/draft_reference_from_local_ocr.py --input-dir "input/data/1" --docling-only
python scripts/draft_reference_from_local_ocr.py --input-dir "input/data/1" --tesseract-only --psm 6
```

---

## MinerU — изображения vs эталон (CLI, в т.ч. Google Colab)

**Зависимости:** `pip install jiwer` + установка [MinerU](https://opendatalab.github.io/MinerU/) (в т.ч. `mineru` в PATH). Метрики те же поля, что в `responses_api_analyze/metrics.py` (CER, Final Score, …). В ноутбуке `notes/statisctics.ipynb` это отдельный **шаг 7b** (`MINERU_INSTALL_MODE` / `MINERU_INSTALL_NOW`), шаг 2 ставит только зависимости GOT-OCR2.

Эталоны рядом с PNG: `stem.ref.txt`, `stem.ref.md`, … Скрипт вызывает `mineru -p <файл> -o <workdir> -b <backend>`, забирает сгенерированный `*.md`, пишет `output/.../hypotheses/mineru/<stem>.md`, JSONL и `mineru_summaries.json`.

**Colab:** Runtime → GPU. Если недоступен Hugging Face: `export MINERU_MODEL_SOURCE=modelscope`. На бесплатном T4 разумно `--backend pipeline`; первый прогон качает модели (долго).

```bash
pip install jiwer
# далее установка mineru по документации проекта
python scripts/mineru_image_benchmark.py \
  --input-dir "input/data/1" \
  --output-dir "output/mineru_benchmark" \
  --backend pipeline \
  --lang cyrillic
```

Скан-подобные страницы (форсировать OCR-ветку пайплайна):

```bash
python scripts/mineru_image_benchmark.py \
  --input-dir "input/data/1" \
  --backend pipeline \
  --method ocr \
  --lang cyrillic
```

Пересчитать CER по уже сохранённым гипотезам (без вызова mineru):

```bash
python scripts/mineru_image_benchmark.py \
  --input-dir "input/data/1" \
  --output-dir "output/mineru_benchmark" \
  --dry-run
```

Полезные переменные окружения: `MINERU_BIN`, `MINERU_BACKEND`, `MINERU_METHOD`, `MINERU_LANG`, `MINERU_API_URL`, `MINERU_FILE_TIMEOUT_SEC`, `MINERU_INPUT_DIR`, `MINERU_OUTPUT_DIR`, `MINERU_NORMALIZE`.

---

## task03 — сравнение эталона и текста (в т.ч. с внешнего OCR)

**Зависимости:** `pip install jiwer`

Сравнение двух файлов UTF-8:

```bash
python scripts/model_analyze/task03_compare_ocr_text_to_reference.py \
  --reference эталон.txt \
  --hypothesis ocr_cloud_output.txt \
  --model "VendorOCR_v1" \
  --pretty
```

С комментарием и подписью парсера:

```bash
python scripts/model_analyze/task03_compare_ocr_text_to_reference.py \
  -r эталон.txt \
  -H ocr_output.txt \
  --model "MyPipeline" \
  --internal-parser "chunk_v2" \
  --comment "Страница 3, скан 300 dpi" \
  --normalize nfkc_ws \
  --pretty
```

Гипотеза из stdin:

```bash
cat ocr_output.txt | python scripts/model_analyze/task03_compare_ocr_text_to_reference.py \
  -r эталон.txt \
  --hypothesis - \
  --model "API_stream" \
  --pretty
```

Строгая нормализация (нижний регистр + пробелы + NFKC):

```bash
python scripts/model_analyze/task03_compare_ocr_text_to_reference.py \
  -r ref.txt -H hyp.txt \
  --model test \
  --normalize nfkc_ws_lower \
  --pretty
```

---

## responses_api_analyze — облачный OCR (SaaS API)

**Зависимости:** `pip install boto3 requests jiwer pillow google-auth google-auth-oauthlib` (см. `notes/requirements-responses-api-colab.txt`).

Провайдеры: `textract`, `google_vision`, `aspose_ocr`, `freeocr`, `freeocr_table` — см. `scripts/responses_api_analyze/README.md`. Локальные движки и свои vLLM не используются.

Amazon Textract:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1
python scripts/responses_api_analyze/run_ocr_api_benchmark.py \
  --providers textract \
  --input-dir "input/data/1"
```

Google Cloud Vision (API key или OAuth `client_secret_….json` + браузер при первом запуске):

```bash
export GOOGLE_VISION_API_KEY=...
# или:
# export GOOGLE_OAUTH_CLIENT_SECRETS_JSON=client_secret_....apps.googleusercontent.com.json
# В консоли Google для Web-клиента добавьте redirect: http://localhost:8085/ (порт = GOOGLE_OAUTH_LOCAL_SERVER_PORT).
python scripts/responses_api_analyze/run_ocr_api_benchmark.py \
  --providers google_vision \
  --input-dir "input/data/1"
```

Aspose.OCR Cloud (или триал `ASPOSE_OCR_USE_TRIAL=1`):

```bash
export ASPOSE_OCR_CLIENT_ID=...
export ASPOSE_OCR_CLIENT_SECRET=...
python scripts/responses_api_analyze/run_ocr_api_benchmark.py \
  --providers aspose_ocr \
  --input-dir "input/data/1"
```

FreeOCR.AI (RapidAPI, текст и таблицы). Язык не задаётся API — только автоопределение (русский в списке поддерживаемых). По умолчанию для `/ocr` передаётся `format=text` (переменная `FREEOCR_OCR_FORMAT`); для таблиц — `FREEOCR_TABLE_FORMAT=json`.

```bash
export FREEOCR_RAPIDAPI_KEY=...
# опционально: export FREEOCR_OCR_FORMAT=json
python scripts/responses_api_analyze/run_ocr_api_benchmark.py \
  --providers freeocr,freeocr_table \
  --input-dir "input/data/1"
```

Все пять сразу (нужны все ключи):

```bash
export API_OCR_USE_ALL=1
export API_OCR_INPUT_DIR=input/data/1
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-east-1
export GOOGLE_VISION_API_KEY=...
export ASPOSE_OCR_CLIENT_ID=... ASPOSE_OCR_CLIENT_SECRET=...
export FREEOCR_RAPIDAPI_KEY=...
python scripts/responses_api_analyze/run_ocr_api_benchmark.py
```

Скрипт при старте сам подхватывает **`.env` в корне репозитория** (уже выставленные в shell переменные не перезаписывает).

Опционально: `API_OCR_GLOB`, `API_OCR_NORMALIZE`, `API_OCR_INTERNAL_PARSER`. Флаги CLI переопределяют env.

---

## Связь с таблицей метрик (кратко)

| Скрипт | Model | CER / Accuracy | Скорость | Токены | Прочее |
|--------|--------|-----------------|----------|--------|--------|
| `run_ocr_api_benchmark` | поле `Model` в `api_ocr_summaries.json` | CER / Accuracy (как task03) | `elapsed_sec` в JSONL | usage при наличии | гипотезы в `hypotheses/<provider>/` |
| task01 | через `--lang` / конфиг в JSON (`tesseract_config`) | нет эталона | `elapsed_sec` в JSONL | нет | прокси таблиц, conf Tesseract |
| task02 | `lang=docling`, пометка в `tesseract_config` | нет эталона | `elapsed_sec` | нет | confidence Docling ≠ слова |
| task03 | `--model` | CER, Accuracy в JSON | нет | грубые chars/words | Final Score = 100×(1−CER) по умолчанию |

При добавлении нового скрипта в `scripts/` **дописывайте сюда секцию** с зависимостями и 1–3 примерами `bash`.
