# Локальные / тяжёлые пайплайны (`model_analyze`)

Здесь лежат скрипты, которые **крутят модели и парсеры локально** (Tesseract, Docling, GOT/DeepSeek через `transformers`, сравнение с эталоном).

| Файл | Назначение |
|------|------------|
| `task01_ocr_tesseract_baseline_metrics.py` | Tesseract, JSONL-метрики |
| `task02_docling_baseline_metrics.py` | Docling |
| `task03_compare_ocr_text_to_reference.py` | CER / Accuracy / Final Score по двум файлам |
| `deepseek_test.py` | пример загрузки DeepSeek-OCR (HF) |

Облачный OCR (Amazon Textract, Google Vision, Aspose.OCR Cloud, FreeOCR.AI RapidAPI) — в `../responses_api_analyze/`.

Запуск из корня репозитория (как раньше, см. `docs/script-run-examples.md` — путь `scripts/model_analyze/...`).
