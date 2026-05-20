#!/usr/bin/env python3
"""
Один вызов на вашей машине: подтягивает переменные из ``.env`` в **корне репозитория**
(как у остальных бенчмарков) и запускает ``yandex_vision_image_benchmark`` с путями по умолчанию.

**Модель на стороне Яндекса** — локально ничего тяжёлого не качается, только ``jiwer`` для метрик
и HTTP к ``recognizeText``.

**``.env``** (пример; не коммитьте ключи)::

  YANDEX_GPT_FOLDER_ID=b1g...
  YANDEX_GPT_API_KEY=AQVN...

Альтернатива: ``YANDEX_OCR_FOLDER_ID`` / ``YANDEX_OCR_API_KEY`` или IAM ``YANDEX_OCR_IAM_TOKEN``
/ ``YC_IAM_TOKEN`` (см. ``scripts/yandex_vision_image_benchmark.py``).

**Запуск** (рабочий каталог = корень репозитория)::

  python scripts/run_yandex_vision_local_once.py

С параметрами Vision (всё остальное — как у основного скрипта)::

  python scripts/run_yandex_vision_local_once.py --model table --language-codes ru,en

Переопределить каталоги::

  python scripts/run_yandex_vision_local_once.py --input-dir other/pngs --output-dir output/yz

Пересчёт CER без API::

  python scripts/run_yandex_vision_local_once.py --dry-run

**Роли и область API-ключа** — см. комментарий в конце этого файла или документацию:
https://yandex.cloud/ru/docs/vision/security/
https://aistudio.yandex.ru/docs/ru/vision/api-ref/authentication
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- Роли Vision OCR (кратко для операторов) ---
# Минимальная сервисная роль на каталог (или выше по иерархии): ai.vision.user
#   см. https://yandex.cloud/ru/docs/vision/security/
# Достаточно и более широких сервисных ролей: ai.editor, ai.admin (включают ai.vision.user).
# IAM-токен пользователя Яндекса: та же роль на каталог + заголовок x-folder-id с id каталога.
# API-ключ сервисного аккаунта: роль ai.vision.user (или шире) на каталог, где создан СА;
#   x-folder-id не передаётся — каталог берётся из привязки ключа к СА
#   (см. https://aistudio.yandex.ru/docs/ru/vision/api-ref/authentication ).
# Область действия (scope) API-ключа: при создании ключа можно задать ограничение
# ``yc.ai.vision.execute`` (только вызовы Vision OCR), см. консоль / ``yc iam api-key create --scopes``.
# Права ключа не заменяют IAM: у субъекта всё равно должна быть роль на каталог (например ai.vision.user).


def _argv_has_flag(argv: list[str], name: str) -> bool:
    prefix = f"{name}="
    for a in argv:
        if a == name or a.startswith(prefix):
            return True
    return False


def _ensure_default_paths() -> None:
    rest = sys.argv[1:]
    if not _argv_has_flag(rest, "--input-dir"):
        sys.argv.extend(["--input-dir", "input/data/1"])
    if not _argv_has_flag(rest, "--output-dir"):
        sys.argv.extend(["--output-dir", "output/yandex_vision"])


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    _ensure_default_paths()
    # При запуске `python scripts/...` первый элемент sys.path — каталог со скриптом (scripts/).
    from yandex_vision_image_benchmark import main as bench_main

    return bench_main()


if __name__ == "__main__":
    raise SystemExit(main())
