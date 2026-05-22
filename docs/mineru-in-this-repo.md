# MinerU в репозитории OCR-Analyze: что, куда и как считается

Этот файл — **описание того, что зашито в коде и доках этого репо**, плюс **состав официального пакета `mineru` на PyPI** для версии **3.1.15** ([раздел 10](#10-версия-пакета-mineru-на-pypi-и-состав-backend-pipeline-3115)). В самом репозитории версия MinerU **не закреплена**: в Colab обычно `pip install -U mineru[pipeline]` — фактическая версия равна последней на момент установки. См. [официальную документацию MinerU](https://opendatalab.github.io/MinerU/).

---

## 1. Роль репозитория

Репозиторий **не встраивает** MinerU как библиотеку с фиксированной моделью. Он:

1. Вызывает **внешний CLI** `mineru` (или путь из `MINERU_BIN`).
2. Забирает сгенерированный **Markdown** (`.md`) из рабочего каталога задачи.
3. Сравнивает с **эталонами** рядом с изображениями и пишет **JSONL**, **сводку JSON**, опционально **склейки гипотез**.

Идентификатор «модели» в метриках — это **строка вида `mineru_cli:<backend>`**, а не один чекпоинт (см. `scripts/mineru_image_benchmark.py`).

---

## 2. Запуск MinerU из этого репо

### Скрипт

| Файл | Назначение |
|------|------------|
| `scripts/mineru_image_benchmark.py` | Прогон по картинкам, вызов `mineru`, метрики, артефакты |

### Команда, которую собирает Python

По сути (детали в коде):

```text
mineru -p <путь_к_изображению> -o <рабочий_каталог_задачи> -b <backend>
```

Опционально добавляются:

- `-m <method>` если задан `--method` / `MINERU_METHOD` (например `ocr` для сканов),
- `-l <lang>` если задан `--lang` / `MINERU_LANG` (в Colab по умолчанию задаётся `cyrillic`),
- `--api-url …` если задан `--api-url` / `MINERU_API_URL` (иначе поднимается временный API, см. docstring скрипта).

### Аргументы и переменные окружения скрипта

Полный список — `python scripts/mineru_image_benchmark.py --help`. Кратко:

| CLI скрипта / env | Значение по умолчанию (если не переопределено) |
|-------------------|--------------------------------------------------|
| `--input-dir` / `MINERU_INPUT_DIR` | `input/data/1` |
| `--glob` | `*.png` |
| `--output-dir` / `MINERU_OUTPUT_DIR` | `output/mineru_benchmark` |
| `--mineru-bin` / `MINERU_BIN` | `mineru` в PATH |
| `--backend` / `MINERU_BACKEND` | `pipeline` |
| `--method` / `MINERU_METHOD` | не передаётся в CLI |
| `--lang` / `MINERU_LANG` | не передаётся в CLI |
| `--api-url` / `MINERU_API_URL` | не передаётся |
| `--timeout` / `MINERU_FILE_TIMEOUT_SEC` | `3600` |
| `--normalize` / `MINERU_NORMALIZE` | `nfkc_ws` |
| `--keep-work` | не удалять `_mineru_work/<stem>` после успеха |
| `--dry-run` | не вызывать `mineru`, только пересчитать метрики по уже лежащим `hypotheses/mineru/<stem>.md` в выбранном `--output-dir` |

### Colab

`notes/mineru_colab.ipynb`: установка `mineru[pipeline]` или `mineru[all]`, затем вызов того же `mineru_image_benchmark.py` с **`OUT_DIR = …/output/mineru_benchmark`**, в ячейке прогона задаются **`BACKEND = "pipeline"`**, **`LANG = "cyrillic"`** (это значения **вашего** ноутбука, не жёсткое требование репо).

---

## 3. Куда пишутся артефакты (важно: два разных места)

### Вариант A — стандартный прогон бенчмарка

`--output-dir` по умолчанию: **`output/mineru_benchmark/`**

| Путь | Содержимое |
|------|------------|
| `…/hypotheses/mineru/<stem>.md` | гипотеза по каждому изображению |
| `…/mineru_runs.jsonl` | по строке на файл: ошибка, время, CER, meta (`backend`, `method`, `lang`, …) |
| `…/mineru_summaries.json` | ключ **`mineru`**: сводка + `_diagnostics` + `_outputs` |
| `…/mineru_hypotheses_raw.json` | при успехе: все гипотезы одним JSON |
| `…/mineru_hypotheses_concat.txt` | при успехе: склейка текстов |
| `…/_mineru_work/<stem>/` | рабочий каталог MinerU на время прогона (удаляется, если не `--keep-work`) |

Поле **`Model`** в сводке = **`mineru_cli:`** + фактический **`backend`** из аргументов (не обязательно слово `pipeline` в таблице, если вы сменили `-b`).

### Вариант B — отчёт `ocr-analyze.odt` и `recalculate_summaries_from_outputs.py`

Скрипты **`fill_ocr_analyze_odt.py`** и **`recalculate_summaries_from_outputs.py`** читают гипотезы из **`output/mineru/<stem>.md`** (плоский каталог **без** `hypotheses/mineru`).

То есть для ODT и быстрого пересчёта сводок нужны файлы именно в **`output/mineru/`**. Если вы гоняли только в `output/mineru_benchmark/`, скопируйте или перенастройте `--output-dir` на `output/mineru`, либо синхронизируйте `.md` вручную.

---

## 4. Эталоны

Порядок поиска эталона рядом с изображением (одинаковая логика в бенчмарке и в `fill_ocr_analyze_odt`):

1. `<stem>.ref.txt`
2. `<stem>.ref.md`
3. `<stem>.txt`
4. `<stem>.md`

Первый непустой файл используется.

---

## 5. Метрики (единый модуль)

Файл: **`scripts/responses_api_analyze/metrics.py`**.

- **Нормализация** по умолчанию **`nfkc_ws`**: NFKC + схлопывание пробелов в один (см. `normalize_text`).
- **CER / WER** — через **jiwer** по нормализованным строкам.
- **Сводная строка** по набору файлов: эталоны и гипотезы **склеиваются через `\n`**, затем один вызов **`build_metric_row`** → один микро-CER/WER на всю склейку (как в `docs/mineru_benchmark_table_snapshot.md`).
- **Final Score** = сумма четырёх частей 0..100, **время не входит** (`composite_final_score`: веса CER 40, WER 35, расхождение длин символов 12.5, слов 12.5). Детали — в `_diagnostics.final_score_parts`.

В **`mineru_runs.jsonl`** по каждому файлу CER считается **отдельно** (для таблицы «по файлам» в ODT).

---

## 6. Отчёт ODT и жёсткая подпись MinerU

`scripts/fill_ocr_analyze_odt.py`:

- Берёт **`output/mineru/*.md`** и **`output/mineru/mineru_runs.jsonl`**.
- В **`build_metric_row`** для сводной строки MinerU передаётся **`model="mineru_cli:pipeline"`** и подпись в таблице **`MinerU (mineru_cli:pipeline)`** — это **фиксированная метка в коде отчёта**, даже если реальный прогон был с другим `-b`. Для строгой воспроизводимости отчёта имеет смысл либо менять код под ваш `backend`, либо всегда гнать `pipeline` для строки «MinerU» в ODT.

---

## 7. Пересчёт без OCR

| Скрипт | MinerU |
|--------|--------|
| `mineru_image_benchmark.py --dry-run` | только при `--output-dir`, где уже есть `hypotheses/mineru/*.md` |
| `recalculate_summaries_from_outputs.py` | читает **`output/mineru/<stem>.md`**, обновляет `output/mineru/mineru_summaries.json`, опционально `--fill-odt` |

---

## 8. Почему в доке разные цифры (Colab vs локально)

`docs/mineru_benchmark_table_snapshot.md`: снимок с **конкретного** прогона Colab; локальные `output/mineru/*.md` могут отличаться байт-в-байт → другой микро-CER и **Final Score**. Это не баг метрик, а **другая гипотеза**.

---

## 9. Где смотреть дальше

- Примеры команд: **`docs/script-run-examples.md`** (секция MinerU).
- Разбор одной строки таблицы и ограничений CER на таблицах: **`docs/mineru_benchmark_table_snapshot.md`**.

---

## 10. Версия пакета MinerU на PyPI и состав backend `pipeline` (3.1.15)

Проверка **2026-05-20**: [PyPI `mineru`](https://pypi.org/project/mineru/) — последняя опубликованная версия **`3.1.15`**. В типичном окружении разработчика репозитория пакет может быть **не установлен** (`pip show mineru` → not found); тогда цифры ниже относятся к **официальному колёсу `mineru-3.1.15`** с PyPI, а не к вашей машине, пока вы явно не поставите эту версию.

### 10.1. Как узнать версию у себя

После установки:

```bash
mineru --version
# или
python -c "import importlib.metadata as m; print(m.version('mineru'))"
```

В Colab зафиксируйте вывод этих команд в отчёте эксперимента или используйте pin: `pip install "mineru[pipeline]==3.1.15"`.

### 10.2. Откуда качаются веса (`pipeline`)

В **`mineru==3.1.15`** модуль `mineru.utils.models_download_utils.auto_download_and_get_model_root_path` при `repo_mode='pipeline'` тянет снапшот репозитория по переменной **`MINERU_MODEL_SOURCE`** (по умолчанию **`huggingface`**):

| `MINERU_MODEL_SOURCE` | Репозиторий (корень снапшота) |
|------------------------|-------------------------------|
| `huggingface` (по умолчанию) | **`opendatalab/PDF-Extract-Kit-1.0`** |
| `modelscope` | **`OpenDataLab/PDF-Extract-Kit-1.0`** |
| `local` | локальный путь из конфигурации `get_local_models_dir()` |

Это же перечислено в **`mineru.utils.enum_class.ModelPath`**: `pipeline_root_hf`, `pipeline_root_modelscope`.

### 10.3. Подпути внутри `PDF-Extract-Kit-1.0` для pipeline (по коду 3.1.15)

Инициализация атомарных моделей — **`mineru.backend.pipeline.model_init`** + **`ModelPath`**:

| Роль | Путь внутри корневого репо | Реализация в коде |
|------|----------------------------|-------------------|
| Layout | `models/Layout/PP-DocLayoutV2` | `PPDocLayoutV2LayoutModel` |
| OCR (общий движок) | `models/OCR/paddleocr_torch` | `PytorchPaddleOCR` (+ `TextSystem` / infer) |
| Таблицы (wired) | `models/TabRec/UnetStructure/unet.onnx` | `UnetTableModel` |
| Таблицы (wireless) | `models/TabRec/SlanetPlus/slanet-plus.onnx` | `PaddleTableModel` |
| Классификация таблицы | `models/TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx` | `PaddleTableClsModel` |
| Формулы (MFR), по умолчанию | `models/MFR/unimernet_hf_small_2503` | `UnimernetModel` |
| Формулы (MFR), альтернатива | `models/MFR/pp_formulanet_plus_m` | `FormulaRecognizer` |

Ветка MFR переключается переменной **`MINERU_FORMULA_CH_SUPPORT`**: `true`/`1`/`yes` → `pp_formulanet_plus_m`, `false`/`0`/`no` или не задано → **`unimernet_small`** (см. `model_init.py`).

### 10.4. Флаг `-l cyrillic` (OCR)

В **`mineru.model.ocr.pytorch_paddle`**: если код языка входит в список `cyrillic_lang` (в т.ч. `ru`, `rs_cyrillic`, `be`, `bg`, `uk`, …), для OCR используется группа языка **`cyrillic`** (единая ветка PaddleOCR-torch внутри скачанного `paddleocr_torch`). Явных имён отдельных ONNX в этом документе нет — они лежат уже **внутри** каталога `models/OCR/paddleocr_torch` снапшота `PDF-Extract-Kit-1.0`.

### 10.5. Backend `vlm` (для полноты, не ваш `-b pipeline`)

В том же `ModelPath` для VLM указано: Hugging Face **`opendatalab/MinerU2.5-Pro-2604-1.2B`**, ModelScope **`OpenDataLab/MinerU2.5-Pro-2604-1.2B`**. При **`mineru -b pipeline`** эти веса **не** подставляются вместо PDF-Extract-Kit.

### 10.6. Extras `mineru[pipeline]` vs `mineru[all]`

В ноутбуке по умолчанию ставится **`mineru[pipeline]`** — меньше зависимостей, режим pipeline. **`mineru[all]`** тянет расширенный набор пакетов (см. метаданные PyPI `requires_dist` для релиза); состав **перечисленных выше** путей к весам для `pipeline` задаётся кодом `model_init`, а не extras.
