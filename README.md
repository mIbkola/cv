# Unitree G1 EDU Ultimate D — Project Workspace

Программно-аппаратный комплекс на базе гуманоидного робота Unitree G1 EDU (Ultimate D).

> ⚠️ Часть деталей не родная (как минимум руки). Учитывать при калибровке манипуляторов и сенсоров усилия.

## Текущие задачи

| ID  | Задача   | Описание                                                                                | Статус     |
|-----|----------|-----------------------------------------------------------------------------------------|------------|
| V   | VOICE    | TTS под голос С. Бурунова + RAG-система для анекдотов 1986 года                          | 🟡 Планирование |
| C   | COFFEE   | Навигация + CV (YOLO/COCO) для поиска стакана кофе, манипуляция с силовой обратной связью, интеграция с LLM/скриптом для команды «принеси кофе Олегу» | 🟡 Планирование |

## Структура репозитория

```
cv/
├── docs/                  # Спеки робота + ТЗ по задачам
│   ├── robot_specs.md     # Тех. характеристики G1 EDU Ultimate D
│   ├── tz_voice.md        # ТЗ: VOICE
│   └── tz_coffee.md       # ТЗ: COFFEE
├── voice/                 # Задача VOICE
│   ├── tts/               # TTS (клон голоса Бурунова)
│   ├── rag/               # RAG по анекдотам 1986
│   └── data/              # Датасеты голоса и текстов
├── coffee/                # Задача COFFEE
│   ├── detection/         # YOLO/COCO детекция стакана
│   ├── manipulation/      # Управление руками + force feedback
│   ├── locomotion/        # Ходьба и навигация
│   └── llm_orchestrator/  # LLM/скрипт-оркестратор «принеси кофе»
├── integration/           # Совместные демо VOICE + COFFEE
└── README.md
```

## Стек (предварительный)

- **Железо:** Unitree G1 EDU Ultimate D, NVIDIA Jetson Orin (опция), RH56DFTP на руках, RealSense D435, Livox MID-360
- **Локомоция:** Unitree SDK (`unitree_sdk2py`), RL-политики (опционально)
- **CV:** YOLOv8 / RT-DETR, COCO pretrained, fine-tune на стаканах
- **TTS:** Silero / XTTS / F5-TTS (клон голоса)
- **RAG:** sentence-transformers + Qdrant/FAISS + LLM
- **Оркестрация:** Python 3.11+, ROS2 (опционально), gRPC/ZeroMQ между модулями

## Quick start

```bash
# клонировать
git clone https://github.com/mIbkola/cv.git
cd cv

# каждый модуль имеет свой README + requirements.txt
# подробности — в docs/
```

## Лицензия

Приватная разработка. Все права защищены.
