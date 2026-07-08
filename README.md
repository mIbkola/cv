# Unitree G1 EDU Ultimate D — Project Workspace

Программно-аппаратный комплекс на базе гуманоидного робота Unitree G1 EDU (Ultimate D).

> ⚠️ Часть деталей не родная (как минимум руки). Учитывать при калибровке манипуляторов и сенсоров усилия.

## Текущие задачи

| ID  | Задача   | Описание                                                                                | Статус     |
|-----|----------|-----------------------------------------------------------------------------------------|------------|
| V   | VOICE    | TTS под голос С. Бурунова + RAG-система для анекдотов 1986 года                          | 🟢 Код написан (mock) |
| C   | COFFEE   | Навигация + CV (YOLO/COCO) для поиска стакана кофе, манипуляция с силовой обратной связью, интеграция с LLM/скриптом для команды «принеси кофе Олегу» | 🟢 Код написан (mock) |

## Структура репозитория

```
cv/
├── docs/                       # Спеки робота + ТЗ по задачам
│   ├── robot_specs.md          # Тех. характеристики G1 EDU Ultimate D
│   ├── tz_voice.md             # ТЗ: VOICE
│   ├── tz_coffee.md            # ТЗ: COFFEE
│   └── dependencies.md         # Источники SDK и драйверов
├── voice/                      # Задача VOICE
│   ├── tts/                    # TTS (клон голоса Бурунова)
│   │   ├── preprocess.py       # Нарезка аудио + транскрипция Whisper
│   │   ├── train.py            # Fine-tune XTTS v2
│   │   └── infer.py            # Инференс TTS
│   ├── rag/                    # RAG по анекдотам 1986
│   │   ├── build_index.py      # Построение индекса Qdrant
│   │   ├── retrieve.py         # Поиск анекдотов
│   │   └── pipeline.py         # Полный пайплайн ASR→intent→RAG→TTS
│   └── data/
│       └── jokes_corpus.jsonl  # 42 анекдота 1986 (10 тем) — стартовый плейсхолдер
├── coffee/                     # Задача COFFEE
│   ├── detection/              # YOLOv8 + RealSense D435 + face_recognition (Олег)
│   │   ├── baseline_test.py    # Тест готовой COCO-модели
│   │   ├── finetune.py         # Fine-tune YOLOv8 на стаканах
│   │   ├── infer.py            # Инференс + 3D-локализация
│   │   └── face_id.py          # Распознавание Олега по лицу
│   ├── manipulation/           # Управление руками + force feedback (RH56DFTP)
│   │   ├── calibrate.py        # Тарировка сенсоров силы
│   │   ├── grasp_controller.py # Force-guided grasp (порог 30–80 г, anti-crush 200 г)
│   │   ├── stability_check.py  # Проверка стабильности хвата
│   │   └── handover.py         # Передача стакана Олегу
│   ├── locomotion/             # Ходьба и навигация
│   │   ├── build_map.py        # Карта через Livox MID-360
│   │   ├── navigator.py        # Навигация (VFH) + SportClient
│   │   └── carry_pose.py       # Поза с грузом + компенсация наклона
│   └── llm_orchestrator/       # Оркестратор «принеси кофе»
│       ├── tools.py            # 7 инструментов (find_object, navigate_to, ...)
│       ├── state_machine.py    # Конечный автомат (11 состояний)
│       └── agent.py            # LLM-агент через vLLM (Qwen2.5-7B)
├── common/                     # Общая инфраструктура
│   ├── config.py               # Конфигурация из YAML
│   ├── logger.py               # rich-логгер
│   ├── state.py                # Модели данных (Detection3D, GraspResult, ...)
│   ├── transport.py            # Обёртка над ZeroMQ (pub/sub + req/rep)
│   ├── mock_hardware.py        # Симулятор G1 для тестов без железа
│   └── monitor.py              # Мониторинг ZMQ-топиков
├── config/
│   └── default.yaml            # Полная конфигурация проекта
├── integration/                # End-to-end демо
│   ├── demo_coffee.py          # Демо: «принеси кофе Олегу» (10 шагов)
│   ├── demo_joke.py            # Демо: «расскажи анекдот 86-го»
│   └── monitor.py              # ZMQ-монитор
├── scripts/
│   └── run_demo.sh             # Запуск mock-демки
├── worklog.md                  # Лог проекта
├── PROGRESS.md                 # Прогресс + открытые вопросы
└── README.md
```

## Стек

- **Железо:** Unitree G1 EDU Ultimate D, NVIDIA Jetson Orin (on-board), внешний GPU-сервер, RH56DFTP, RealSense D435, Livox MID-360
- **Локомоция:** `unitree_sdk2py` (SportClient),VFH-обход препятствий
- **CV:** YOLOv8 (baseline → fine-tune), RealSense D435 для 3D, face_recognition для Олега
- **TTS:** XTTS v2, fine-tune на голосе Бурунова
- **RAG:** sentence-transformers + Qdrant + LLM-ранкер
- **LLM:** Qwen2.5-7B-Instruct локально через vLLM
- **Оркестрация:** Python 3.11+, ZeroMQ (pub/sub + req/rep), `transitions` для state machine

## Быстрый старт (mock-режим, без железа)

```bash
# Установить зависимости
pip install pyzmq pydantic rich click pyyaml transitions

# Демо «принеси кофе Олегу» — 10 шагов, ~2 сек
python integration/demo_coffee.py

# Демо «расскажи анекдот 86-го про штирлица»
python integration/demo_joke.py --query "анекдот про штирлица"

# Полный ZMQ-режим: MockG1 как сервер + demo_coffee как клиент
bash scripts/run_demo.sh

# State machine напрямую
python coffee/llm_orchestrator/state_machine.py --command "принеси кофе Олегу"

# Мониторинг ZMQ-топиков (в другом терминале во время демки)
python common/monitor.py
```

## Что требует датасетов / железа / внешних сервисов

| Компонент | Что нужно |
|---|---|
| `voice/tts/preprocess.py` + `train.py` | Аудио/видео с голосом Бурунова (≥10 мин) |
| `voice/rag/build_index.py` | Запущенный Qdrant: `docker run -d -p 6333:6333 qdrant/qdrant` |
| `voice/rag/pipeline.py` (LLM) | Локальный vLLM с Qwen2.5-7B-Instruct |
| `coffee/detection/baseline_test.py` + `finetune.py` | Датасет фото бумажных стаканов (200–500 шт.) |
| `coffee/detection/infer.py --live` | RealSense D435 + ultralytics |
| `coffee/detection/face_id.py --live` | Фото Олега + RealSense D435 |
| `coffee/manipulation/*.py` (real) | Unitree G1 + RH56DFTP + `unitree_sdk2py` |
| `coffee/locomotion/*.py` (real) | Unitree G1 + Livox MID-360 |
| `coffee/llm_orchestrator/agent.py --llm qwen-local` | vLLM с Qwen2.5-7B |

## Лицензия

Приватная разработка. Все права защищены.
