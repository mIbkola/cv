# Лог проекта

## 2026-07-09 — Инициализация

### Сделано
- Создан репо `mIbkola/cv` (public)
- Структура:
  - `docs/` — robot_specs.md, tz_voice.md, tz_coffee.md, dependencies.md
  - `voice/` — tts/, rag/, data/
  - `coffee/` — detection/, manipulation/, locomotion/, llm_orchestrator/
  - `integration/`
- Каждая папка имеет README + requirements.txt

### Open questions (ждём ответа)
1. Есть ли доступ к железу уже сейчас, или только планируем?
2. Какой compute: Jetson Orin on-board или внешний сервер?
3. Голос Бурунова — где брать датасет (есть ли уже заготовки)?
4. Анекдоты 1986 — есть ли готовый корпус или собирать?
5. Олег — реальный человек в комнате, или надо детекцию по лицу / голосу?
6. Доступ к GPT API есть? Или локальный LLM?
7. ROS2 или чистый Python + ZeroMQ?

### Следующие шаги
- Ответить на open questions
- Начать с COFFEE-detection baseline test (быстрее всего)
- Параллельно собрать датасет голоса Бурунова
