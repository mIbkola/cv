# Coffee — LLM Orchestrator

Оркестратор: принимает команду «принеси кофе Олегу», разбирает её на subtasks, координирует остальные модули (detection, manipulation, locomotion).

## Подход: гибрид LLM + state machine

LLM (или простой парсер) извлекает интент и параметры:
- `action: "deliver_item"`
- `item: "coffee"`
- `target_person: "Олег"`

Дальше — детерминированная стейт-машина, вызывающая tools из других модулей.

## Стек
- Python 3.11+
- `openai` (если GPT-4o-mini) или `vllm` (локальный Qwen2.5-7B-Instruct)
- `pydantic` для схем
- `transitions` для state machine
- gRPC / ZeroMQ для связи с другими модулями (опционально)

## Установка
```bash
cd coffee/llm_orchestrator
pip install -r requirements.txt
```

## State Machine
```
IDLE
  ↓ (команда "принеси кофе Олегу")
PARSE_CMD
  ↓
FIND_CUP (CV: detection.find_object("cup"))
  ↓ (получили 3D координаты)
APPROACH_CUP (locomotion.navigate_to)
  ↓ (дошли, в 0.5 м от стакана)
GRASP (manipulation.grasp_with_force_feedback)
  ↓
STABILITY_CHECK (manipulation.verify_grasp)
  ├─ fail → RETRY (до 3 раз) → FAILED → сообщить голосом
  └─ ok ↓
FIND_TARGET (CV: detection.find_person("Олег"))
  ↓
APPROACH_TARGET (locomotion.navigate_to)
  ↓ (в 0.7 м от Олега)
HANDOVER (manipulation.handover)
  ↓ (сила упала < 20 г → стакан принят)
RELEASE
  ↓
IDLE
```

## LLM Tools
```python
tools = [
    "find_object(class_name) -> {x, y, z}",
    "navigate_to(x, y) -> bool",
    "grasp_with_force_feedback(target_pose) -> bool",
    "verify_grasp() -> {stable: bool, force: float}",
    "find_person(name) -> {x, y, z}",
    "handover() -> bool",
    "speak(text) -> None  # через voice/tts",
]
```

## Использование
```bash
# Скрипт-режим (без LLM)
python state_machine.py --command "принеси кофе Олегу"

# LLM-режим (GPT-4o-mini)
python agent.py --command "принеси кофе Олегу" --llm gpt-4o-mini
```

## Интеграция с VOICE
- Команда может прийти голосом → ASR → orchestrator
- Статусные сообщения: «Иду за кофе», «Не нашёл стакан», «Олег, держи» → через TTS Бурунова

## TODO
- [ ] Определить интерфейсы tools (контракты между модулями)
- [ ] Реализовать `state_machine.py` с базовым потоком
- [ ] Реализовать `agent.py` с LLM function calling
- [ ] Добавить обработку ошибок и retry-логику
- [ ] Подключить голосовой ввод/вывод
