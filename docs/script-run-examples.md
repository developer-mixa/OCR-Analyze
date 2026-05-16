# Примеры запуска скриптов

Команды ниже предполагают **текущий каталог = корень репозитория** (`AnalytiscNeuroOcr`).

---

## task01 — Tesseract, метрики по изображениям

**Зависимости:** `pip install pillow pytesseract` + системный `tesseract` и языки (`rus`, `eng` и т.д.).

Базовый прогон по умолчанию (`input/data/1`, `*.png`), JSONL + сводка по-русски:

```bash
python scripts/task01_ocr_tesseract_baseline_metrics.py \
  --input-dir "input/data/1" \
  --output-jsonl output/task01_tesseract_metrics.jsonl \
  --print-summary \
  --print-summary-ru
```

Только JSON-сводка:

```bash
python scripts/task01_ocr_tesseract_baseline_metrics.py --input-dir "input/data/1" --print-summary
```

Другие языки и режимы Tesseract:

```bash
python scripts/task01_ocr_tesseract_baseline_metrics.py \
  --input-dir "input/data/1" \
  --lang "rus+eng" \
  --oem 3 \
  --psm 6 \
  --output-jsonl output/task01_tesseract_metrics.jsonl
```

Увеличить изображение перед OCR:

```bash
python scripts/task01_ocr_tesseract_baseline_metrics.py \
  --input-dir "input/data/1" \
  --scale 1.5 \
  --output-jsonl output/task01_tesseract_scale15.jsonl \
  --print-summary-ru
```

---

## task02 — Docling, те же поля метрик (другая семантика confidence)

**Зависимости:** `pip install docling pillow` (установка тяжёлая).

```bash
python scripts/task02_docling_baseline_metrics.py \
  --input-dir "input/data/1" \
  --glob "*.png" \
  --output-jsonl output/task02_docling_metrics.jsonl \
  --print-summary \
  --print-summary-ru
```

PDF (если положить файлы в каталог и сменить glob):

```bash
python scripts/task02_docling_baseline_metrics.py \
  --input-dir "input/data/pdf" \
  --glob "*.pdf" \
  --output-jsonl output/task02_docling_pdf.jsonl \
  --print-summary-ru
```

Метка пайплайна в поле `tesseract_config` (префикс `docling:`):

```bash
python scripts/task02_docling_baseline_metrics.py \
  --input-dir "input/data/1" \
  --pipeline-note "DocumentConverter(default)+manual" \
  --output-jsonl output/task02_docling_metrics.jsonl
```

---

## task03 — сравнение эталона и текста (в т.ч. с внешнего OCR)

**Зависимости:** `pip install jiwer`

Сравнение двух файлов UTF-8:

```bash
python scripts/task03_compare_ocr_text_to_reference.py \
  --reference эталон.txt \
  --hypothesis ocr_cloud_output.txt \
  --model "VendorOCR_v1" \
  --pretty
```

С комментарием и подписью парсера:

```bash
python scripts/task03_compare_ocr_text_to_reference.py \
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
cat ocr_output.txt | python scripts/task03_compare_ocr_text_to_reference.py \
  -r эталон.txt \
  --hypothesis - \
  --model "API_stream" \
  --pretty
```

Строгая нормализация (нижний регистр + пробелы + NFKC):

```bash
python scripts/task03_compare_ocr_text_to_reference.py \
  -r ref.txt -H hyp.txt \
  --model test \
  --normalize nfkc_ws_lower \
  --pretty
```

---

## Связь с таблицей метрик (кратко)

| Скрипт | Model | CER / Accuracy | Скорость | Токены | Прочее |
|--------|--------|-----------------|----------|--------|--------|
| task01 | через `--lang` / конфиг в JSON (`tesseract_config`) | нет эталона | `elapsed_sec` в JSONL | нет | прокси таблиц, conf Tesseract |
| task02 | `lang=docling`, пометка в `tesseract_config` | нет эталона | `elapsed_sec` | нет | confidence Docling ≠ слова |
| task03 | `--model` | CER, Accuracy в JSON | нет | грубые chars/words | Final Score = 100×(1−CER) по умолчанию |

При добавлении нового скрипта в `scripts/` **дописывайте сюда секцию** с зависимостями и 1–3 примерами `bash`.
