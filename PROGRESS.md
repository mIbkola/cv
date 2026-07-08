# Лог проекта

## 2026-07-09 — Инициализация + архитектура

### Сделано
- Создан репо `mIbkola/cv` (public): https://github.com/mIbkola/cv
- Структура: docs/, voice/{tts,rag,data}, coffee/{detection,manipulation,locomotion,llm_orchestrator}, common/, integration/, scripts/, config/
- Все README на русском, requirements.txt на русском
- Архитектура: гибридный compute (CV/манипуляция на Jetson Orin, TTS/LLM/RAG на внешнем GPU), Python+ZeroMQ, локальный Qwen2.5-7B

### Решения
- Транспорт: ZeroMQ (pub/sub + req/rep)
- LLM: локальный Qwen2.5-7B-Instruct через vLLM
- TTS: XTTS v2, fine-tune на голосе Бурунова
- CV: YOLOv8 (baseline → fine-tune), при отсутствии — лёгкий OpenCV-детектор
- Олег: распознавание по лицу (face_recognition)

## 2026-07-09 — Фаза 1: написан код всех модулей (mock-режим)

3 агента параллельно:
- **VOICE** (Task V): TTS preprocess/train/infer + RAG build_index/retrieve/pipeline + 42 анекдота в корпусе
- **COFFEE** (Task C): detection (YOLO+face_id) + manipulation (force feedback) + locomotion + llm_orchestrator (state_machine+agent) — 18 файлов
- **COMMON+INTEGRATION** (Task I): common/{config,logger,state,transport,mock_hardware,monitor}.py + config/default.yaml + integration/{demo_coffee,demo_joke,monitor}.py

Smoke-тесты пройдены: demo_coffee 10/10 за 2.2 сек, demo_joke находит анекдот.

## 2026-07-09 — Фаза 2: реальные данные + лёгкие fallback'ы

### Проблема
- sentence-transformers и ultralytics (YOLOv8) не ставятся: тянут torch, на диске 9.9 ГБ, нет места для зависимостей
- yt-dlp отсутствует — нельзя скачать аудио Бурунова автоматически

### Решение: лёгкие fallback'ы
2 агента параллельно:

**VOICE-EXPAND (Task V2):**
- Корпус анекдотов расширен с 42 → **160 шт.** (12 тем: штирлиц 20, чукча 15, магазин 15, лечащий врач 12, василий иванович 10, поручик ржевский 10, старые евреи 10, участковый 10, брежнев 10, школа 10, студенты 10, автомобиль/ГАИ 10, + 18 прочее)
- Создан `voice/rag/retrieve_light.py` — лёгкий RAG на sklearn TF-IDF + cosine + русский стемминг (nltk), БЕЗ torch
- Обновлён `voice/rag/pipeline.py` — 3-уровневый fallback: Qdrant+ST → retrieve_light → keyword
- Обновлён `integration/demo_joke.py` — 3-уровневый RAG, пометки источника в отчёте
- Тесты пройдены: retrieve_light находит релевантные анекдоты за 1-3 мс

**COFFEE-DATA (Task C2):**
- Создан `scripts/gen_synthetic_cups.py` — генератор 50 синтетических фото бумажных стаканов (PIL+numpy, разные фоны/ракурсы/sleeve/крышка/пар)
- Создан `coffee/detection/baseline_light.py` — детектор на чистом OpenCV (HSV-сегментация + морфология + findContours + edge-fallback + NMS), БЕЗ нейросетей
- Сгенерировано 50 изображений + ground truth (87 стаканов)
- **Precision = 0.981, Recall = 0.609, F1 = 0.752** (IoU ≥ 0.3)
- Ограничение: recall падает когда цвет фона совпадает с цветом стакана (бежевый на бежевом) — нужен fine-tune YOLO на реальных фото для продакшна
- Аннотированные изображения сохранены в `annotated/`

### Что работает сейчас БЕЗ железа, БЕЗ тяжёлых моделей
```bash
cd /home/z/my-project/work/cv

# Анекдоты — реальный TF-IDF поиск по корпусу 160 шт.
/home/z/.venv/bin/python3 voice/rag/retrieve_light.py --query "анекдот про штирлица" --top-k 3
/home/z/.venv/bin/python3 integration/demo_joke.py --query "расскажи анекдот 86 года про чукчу"

# CV — реальный OpenCV-детектор на синтетике
/home/z/.venv/bin/python3 scripts/gen_synthetic_cups.py --count 50
/home/z/.venv/bin/python3 coffee/detection/baseline_light.py --images coffee/detection/test_images/ --visualize

# Mock-демки (полные сценарии)
/home/z/.venv/bin/python3 integration/demo_coffee.py
/home/z/.venv/bin/python3 coffee/llm_orchestrator/state_machine.py --command "принеси кофе Олегу"
```

### TODO (нужны внешние ресурсы)
- [ ] Аудио/видео с голосом Бурунова (≥10 мин) → для fine-tune XTTS. yt-dlp нет, пользователь должен скинуть файлы
- [ ] Реальные фото бумажных стаканов (200-500 шт.) → для fine-tune YOLOv8. Пока только синтетика
- [ ] Фото лица Олега → для face_id
- [ ] Запуск Qdrant через Docker (или локально) для векторного RAG
- [ ] Запуск vLLM с Qwen2.5-7B-Instruct на GPU-сервере для LLM-оркестратора

### Следующие шаги
1. Пользователь кидает аудио Бурунова → запускаем preprocess + train
2. Пользователь кидает фото стаканов → запускаем baseline_test (если ultralytics встанет) или размечаем для fine-tune
3. Пользователь кидает своё лицо (или фото Олега) → тестируем face_id
4. На GPU-сервере поднимаем vLLM + Qwen → тестируем agent.py
5. На железе G1 — переключаем --mock на --real в скриптах манипуляции/локомоции
