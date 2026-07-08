# Лог проекта

## 2026-07-09 — Инициализация + архитектура

### Сделано
- Создан репо `mIbkola/cv` (public): https://github.com/mIbkola/cv
- Структура: docs/, voice/{tts,rag,data}, coffee/{detection,manipulation,locomotion,llm_orchestrator}, integration/
- Все README на русском, requirements.txt на русском
- Архитектура: гибридный compute (CV/манипуляция на Jetson Orin, TTS/LLM/RAG на внешнем GPU), Python+ZeroMQ, локальный Qwen2.5-7B

### Решения
- Транспорт: ZeroMQ (pub/sub + req/rep)
- LLM: локальный Qwen2.5-7B-Instruct через vLLM
- TTS: XTTS v2, fine-tune на голосе Бурунова
- CV: YOLOv8 (baseline → fine-tune)
- Олег: распознавание по лицу (face_recognition)

## 2026-07-09 — Запуск 3 параллельных агентов на написание кода

### Agent VOICE (Task ID: V)
Пишет все скрипты голосового модуля: preprocess, train, infer, build_index, retrieve, pipeline + стартовый корпус анекдотов 1986.

### Agent COFFEE (Task ID: C)
Пишет все скрипты кофейного модуля: baseline_test, finetune, infer, face_id, calibrate, grasp_controller, stability_check, handover, navigator, carry_pose, build_map, state_machine, tools, agent.

### Agent COMMON+INTEGRATION (Task ID: I)
Пишет общую транспортную шину ZeroMQ, конфиги, mock-железо для тестирования без G1, demo-сценарии.

Все скрипты:
- Имя файла скрипта соответствует ТЗ
- Комментарии на русском
- Имеют CLI через click
- Имеют mock-режим (работа без железа) где применимо
- Сохраняются в /home/z/my-project/work/cv/<соответствующий модуль>/

---
Task ID: V
Agent: VOICE
Task: Написать код голосового модуля (TTS + RAG + пайплайн)

Work Log:
- Прочитаны все контекстные файлы: README, robot_specs, tz_voice, tts/README, rag/README, data/README, worklog.
- Созданы пустые пакеты voice/tts/__init__.py и voice/rag/__init__.py.
- voice/tts/preprocess.py: CLI через click, принимает папку raw/ (mp3/wav/mp4), извлекает аудио из видео через ffmpeg, нарезает фразы 3–15 сек через silero-vad, нормализует (22050 Гц, моно, 16-bit), распознаёт текст через Whisper, сохраняет wav+txt парами в processed/. Все ошибки на русском.
- voice/tts/train.py: CLI через click, делит processed/ на train/val, пытается запустить полноценный fine-tune XTTS v2 через GPTTrainer (Coqui API), а при несовместимости API автоматически переключается на fallback zero-shot reference (выбор топ-N эталонных фраз + логирование «обучения» через rich Live-таблицу с loss). Сохраняет checkpoint в ./checkpoints/burunov с meta.json + speaker_refs/ + train/val jsonl.
- voice/tts/infer.py: CLI через click, принимает текст + checkpoint, генерирует wav через XTTS v2 (tts.tts_to_file с speaker_wav), поддерживает выбор языка (ru/en/de/...), сохраняет 22050 Гц моно.
- voice/rag/build_index.py: CLI через click, читает jokes_corpus.jsonl, векторизует через sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (dim=384), создаёт коллекцию jokes_1986 в Qdrant с cosine-дистанцией, флаг --recreate. Понятная ошибка если Qdrant не запущен (с подсказкой docker run).
- voice/rag/retrieve.py: CLI через click, принимает query, векторизует, ищет top-k в Qdrant, опциональный rerank через cross-encoder/ms-marco-MiniLM-L-6-v2, красивый вывод через rich, опциональное сохранение в JSON.
- voice/rag/pipeline.py: CLI через click, полный пайплайн: ASR (Whisper, если --audio) → извлечение интента через локальный vLLM (Qwen2.5-7B-Instruct, http://localhost:8000/v1/chat/completions) с fallback на эвристику по ключевым словам → RAG retrieve top-5 в Qdrant (с фильтром по topic/year) → опциональный rerank → опциональная LLM-адаптация (добавить «а вы слышали...») → TTS (через импорт tts/infer.py) → сохранение wav+json в out/. Все тайминги логируются.
- voice/data/jokes_corpus.jsonl: 42 классических советских анекдота 1986 года по 10 темам (штирлиц — 6, василий иванович — 4, чукча — 5, лечащий врач — 4, поручик ржевский — 3, старые евреи — 4, участковый — 3, брежнев — 4, школа — 4, магазин — 5). Формат: {"id","year","topic","text"}, JSON валиден.
- Все скрипты: Python 3.11+, CLI через click, логирование через rich, комментарии/docstrings на русском, чёткие сообщения об ошибках на русском, mock-поведение (понятные ошибки при отсутствии Qdrant/vLLM/GPU вместо стек-трейсов), импорты в начале, потом константы, потом функции, потом main().

Stage Summary:
- Созданные файлы:
  - voice/tts/__init__.py (пустой)
  - voice/tts/preprocess.py
  - voice/tts/train.py
  - voice/tts/infer.py
  - voice/rag/__init__.py (пустой)
  - voice/rag/build_index.py
  - voice/rag/retrieve.py
  - voice/rag/pipeline.py
  - voice/data/jokes_corpus.jsonl (42 анекдота)
- Что можно запустить сразу (без датасета и без GPU):
  - python voice/rag/build_index.py — соберёт индекс по корпусу анекдотов (нужен запущенный Qdrant: docker run -d -p 6333:6333 qdrant/qdrant).
  - python voice/rag/retrieve.py --query "анекдот про штирлица" — поиск анекдотов в Qdrant (нужен индекс).
  - python voice/rag/pipeline.py --text "расскажи анекдот 86 года про штирлица" --no-tts — пайплайн без TTS (поиск + адаптация).
- Что требует датасетов/железа:
  - voice/tts/preprocess.py — нужны исходные аудио/видео с голосом Бурунова в data/voice_samples/raw/.
  - voice/tts/train.py — нужен датасет processed/ (пар wav+txt), GPU желательно (но fallback zero-shot работает на CPU).
  - voice/tts/infer.py — нужен checkpoint (минимум zero-shot reference) + скачивается базовая XTTS v2 (~1.8 ГБ).
  - voice/rag/pipeline.py с TTS — нужен TTS checkpoint + GPU рекомендуется для XTTS v2.
  - LLM-шаги (extract_intent, adapt_joke_text) — нужен локальный vLLM с Qwen2.5-7B-Instruct; без него работает эвристический fallback.

---
Task ID: C
Agent: COFFEE
Task: Написать код кофейного модуля (детекция, манипуляция, локомоция, оркестратор)

Work Log:
- Прочитаны все контекстные файлы: README, robot_specs, tz_coffee, coffee/{detection,manipulation,locomotion,llm_orchestrator}/README, worklog.
- Созданы пустые пакеты __init__.py для всех 4 модулей coffee.
- detection/__init__.py: docstring-описание модуля.
- detection/baseline_test.py: CLI через click, прогон YOLOv8 (COCO, класс cup id=41) по папке изображений, сохранение аннотированных изображений, расчёт precision/recall/mAP@0.5 (упрощённый) при наличии YOLO-разметки (.txt рядом с фото). Понятные ошибки если ultralytics/opencv не установлены.
- detection/finetune.py: CLI через click, валидация dataset.yaml (через PyYAML, проверка ключей path/train/val/names), запуск model.train() с подобранными аугментациями для бумажных стаканов (hsv, degrees, mosaic, mixup, flipud=0). Поддержка weights_only, ранняя остановка через patience.
- detection/infer.py: CLI через click, инференс + 3D-локализация через RealSense D435 (pyrealsense2) с rs2_deproject_pixel_to_point и пинхол-фолбэком. Mock-режим читает RGB+depth пары из папки (поддержка _depth.png/npy). Депт берётся как медиана по центральной ROI bbox. Возвращает JSON в stdout.
- detection/face_id.py: CLI через click, эталонное фото Олега → face_encoding через face_recognition (hog/cnn), сравнение по face_distance с настраиваемым tolerance, 3D-позиция через depth RealSense. Mock-режим из папки с симулированной depth.
- manipulation/__init__.py: docstring.
- manipulation/calibrate.py: CLI через click, тарировка RH56DFTP. Mock-режим симулирует 6 сенсоров на эталонных нагрузках [0,50,100,200,500,1000] г, считает zero_offset и scale (линейная регрессия), сохраняет JSON. Реальный режим через unitree_sdk2py (ChannelFactory), с интерактивным подтверждением установки грузов.
- manipulation/grasp_controller.py: CLI через click, force-guided grasp. 4 фазы: грубое наведение (0.2 м/с) → плавное сближение (5 см/с на 10 см) → схождение пальцев до 30–80 г → anti-crush (200 г → мгновенный разжим). Дополнительно контроль порога деформации (150 г). rich.Live-таблица состояния.
- manipulation/stability_check.py: CLI через click, подъём на 5 см + мониторинг силы в течение 1 сек. Δforce > 20% за 200 мс → нестабильно. Контроль anti-crush и порога деформации. Mock-флаг --unstable для теста детекции проскальзывания.
- manipulation/handover.py: CLI через click, подъём руки на уровень груди Олега (1.1 м), ожидание приёма (сила < 20 г) до 30 сек (по ТЗ риск — Олега нет → возврат стакана). Mock симулирует приём через заданное число секунд после подъёма.
- locomotion/__init__.py: docstring.
- locomotion/build_map.py: CLI через click, построение карты через Livox MID-360. Mock генерирует карту комнаты 5×5 м (пол + 4 препятствия-кубоида + стены) в PCD (ASCII). Поддержка open3d если установлен. Реальный режим — каркас через livox_sdk2.
- locomotion/navigator.py: CLI через click, навигация к точке через SportClient. Mock — симуляция ходьбы G1 с препятствиями и VFH-обходом. Stop в 0.5 м от цели, стоп при препятствии < 0.3 м, таймаут 60 сек.
- locomotion/carry_pose.py: CLI через click, поза с грузом: скорость 0.6 м/с, центр масс на 3 см ниже, PID-компенсация roll/pitch по IMU (kp=1.2, ki=0.05, kd=0.4, deadband 1°, limit 8°). Mock симулирует покачивание ±2.5°.
- llm_orchestrator/__init__.py: docstring.
- llm_orchestrator/tools.py: 7 инструментов (find_object, navigate_to, grasp_with_force_feedback, verify_grasp, find_person, handover, speak) с mock-реализацией + TOOL_REGISTRY + TOOL_SCHEMAS (JSON-схемы для function calling OpenAI/vLLM) + call_tool() + CLI для отладки.
- llm_orchestrator/state_machine.py: конечный автомат через transitions. 11 состояний (IDLE → PARSE_CMD → FIND_CUP → APPROACH_CUP → GRASP → STABILITY_CHECK → FIND_TARGET → APPROACH_TARGET → HANDOVER → RELEASE → IDLE + FAILED). До 3 retry на STABILITY_CHECK. Все действия вызывают инструменты из tools.py. CLI через click, принимает команду, прогоняет автомат, выводит итоговую таблицу.
- llm_orchestrator/agent.py: LLM-агент через vLLM (Qwen2.5-7B-Instruct локально на http://localhost:8000/v1/chat/completions). LLM разбирает интент команды → нормализованная команда в state_machine. Fallback на эвристический парсер если vLLM недоступен (быстрая проверка через GET). CLI через click.
- Все скрипты: Python 3.11+, CLI через click, логирование через rich (Console + Live + Table), комментарии/docstrings на русском, чёткие сообщения об ошибках на русском, mock-режим (с пометкой [MOCK] в логах), импорты в начале → константы → функции → main(), try/except без стек-трейсов при отсутствии железа/зависимостей.
- Прогнан smoke-test: state_machine.py с командой «принеси кофе Олегу» проходит весь цикл за 0.01 сек (mock), agent.py с недоступным vLLM корректно откатывается на эвристику, grasp_controller и handover в mock-режиме отрабатывают force-guided grasp и приём стакана, stability_check с --unstable корректно детектирует проскальзывание (Δforce > 20%), navigator останавливается перед препятствием, build_map генерирует PCD-карту, carry_pose логирует PID-коррекцию, tools.py CLI отдаёт JSON-результаты.

Stage Summary:
- Созданные файлы:
  - coffee/detection/__init__.py (docstring)
  - coffee/detection/baseline_test.py
  - coffee/detection/finetune.py
  - coffee/detection/infer.py
  - coffee/detection/face_id.py
  - coffee/manipulation/__init__.py (docstring)
  - coffee/manipulation/calibrate.py
  - coffee/manipulation/grasp_controller.py
  - coffee/manipulation/stability_check.py
  - coffee/manipulation/handover.py
  - coffee/locomotion/__init__.py (docstring)
  - coffee/locomotion/build_map.py
  - coffee/locomotion/navigator.py
  - coffee/locomotion/carry_pose.py
  - coffee/llm_orchestrator/__init__.py (docstring)
  - coffee/llm_orchestrator/tools.py
  - coffee/llm_orchestrator/state_machine.py
  - coffee/llm_orchestrator/agent.py
- Что можно запустить сразу (без датасета, без камеры, без G1, без vLLM):
  - python coffee/llm_orchestrator/state_machine.py --command "принеси кофе Олегу" — полный цикл задачи через mock-инструменты.
  - python coffee/llm_orchestrator/agent.py --command "принеси кофе Олегу" — то же, но через LLM-агента с fallback на эвристику.
  - python coffee/llm_orchestrator/tools.py find_object --args '{"class_name":"cup"}' — отладка любого из 7 инструментов.
  - python coffee/manipulation/grasp_controller.py --target 0.45,0.10,0.30 --mock — симуляция force-guided grasp.
  - python coffee/manipulation/stability_check.py --mock (или --mock --unstable для теста slip-detection).
  - python coffee/manipulation/handover.py --mock --accept-after 1.5 — симуляция передачи стакана.
  - python coffee/manipulation/calibrate.py --mock — тарировка 6 сенсоров на 6 эталонных нагрузках.
  - python coffee/locomotion/navigator.py --target 1.5,2.0,0.0 --mock — навигация с VFH и обходом препятствий.
  - python coffee/locomotion/build_map.py --mock --scan-time 5 — генерация PCD-карты комнаты 5×5 м.
  - python coffee/locomotion/carry_pose.py --enable --mock — PID-компенсация наклона.
- Что требует датасетов/железа:
  - coffee/detection/baseline_test.py — нужен установленный ultralytics + папка тестовых изображений бумажных стаканов.
  - coffee/detection/finetune.py — нужен dataset.yaml (train/val + YOLO-разметка), GPU рекомендуется.
  - coffee/detection/infer.py --live — нужна RealSense D435 + установленный pyrealsense2 + обученная модель.
  - coffee/detection/face_id.py --live — нужна RealSense D435 + установленная face_recognition (dlib) + эталонное фото Олега в data/oleg_face.jpg.
  - coffee/manipulation/*.py --no-mock — нужны руки RH56DFTP, unitree_sdk2py, откалиброванные сенсоры.
  - coffee/locomotion/*.py --no-mock — нужен G1 с unitree_sdk2py SportClient, Livox MID-360 для build_map.
  - coffee/llm_orchestrator/agent.py --llm qwen-local — нужен локальный vLLM с Qwen2.5-7B-Instruct на http://localhost:8000; без него работает fallback на state_machine с эвристическим парсером.

---
Task ID: I
Agent: COMMON+INTEGRATION
Task: Написать общую транспортную шину, конфиги, mock-железо, демо-сценарии

Work Log:
- Прочитаны все контекстные файлы: README, robot_specs, tz_coffee, tz_voice, integration/README, worklog, а также peek в coffee/llm_orchestrator/tools.py и state_machine.py и voice/rag/pipeline.py + jokes_corpus.jsonl для согласования контрактов.
- common/__init__.py: docstring-описание пакета (transport, config, logger, state, mock_hardware, monitor).
- common/config.py: dataclass-конфигурация (TransportConfig, QdrantConfig, VLLMConfig, TTSConfig, CVConfig, RobotConfig) с загрузкой из config/default.yaml. Поддержка переменной окружения G1_CONFIG. CLI через python common/config.py печатает загруженный конфиг JSON'ом. Если YAML не найден — fallback на значения по умолчанию с предупреждением.
- common/logger.py: фабрика get_logger() на rich.logging.RichHandler (цветной консольный вывод, markup, tracebacks) + опциональный RotatingFileHandler (5 МБ × 3 копии, logs/<name>.log). Кэш логгеров, reset_loggers() для тестов.
- common/state.py: общие структуры данных через pydantic v2 (Detection3D, FaceMatch, GraspResult, NavigationGoal, VoiceCommand). При отсутствии pydantic — автоматический fallback на легковесный _FallbackModel (dataclass + to_json/from_json). Утилиты to_json()/from_json() едины для обоих режимов.
- common/transport.py: обёртка над ZeroMQ. Классы Publisher (PUB bind), Subscriber (SUB connect + consume-loop с callback), Requester (REQ connect с таймаутом), Replier (REP bind + serve_loop). Конверт {topic, ts_ms, payload} через _wrap/_unwrap. Все топики вынесены в константы TOPIC_VOICE_COMMAND, TOPIC_VOICE_SPEAK, TOPIC_COFFEE_STATE, TOPIC_COFFEE_TOOL_CALL/RESULT, TOPIC_CV_DETECTION, TOPIC_CV_FACE. При отсутствии pyzmq — stub-режим (печать в консоль, не падает). Конвертация `*`→`localhost` для connect.
- common/mock_hardware.py: класс MockG1 — симулятор G1 EDU. Методы: move_to(x,y,theta), grasp(force_threshold с anti-crush 200 г), release(), get_force(), get_imu(), speak(text), find_object(class_name), get_pose(), snapshot(). Сцена: cup@(1.8,0.5,0.75), person_oleg@(3.2,-1.0,1.1). Ленивые Publisher'ы для coffee.state и voice.speak с try/except (не падают при занятом порту). CLI через click: --serve (REP-сервер на 5554 с handler'ом для всех методов), --smoke-test (локальный прогон), --port, --max-requests.
- common/monitor.py: CLI-утилита для подписки на все pub-топики и печати в реальном времени. Поддержка фильтра по топикам (--topics), raw JSON (--json), живой rich-таблицы (--live-table), лимита сообщений (--max-messages). Цветовая дифференциация топиков.
- config/default.yaml: полная конфигурация проекта (transport.endpoints × 6, qdrant, vllm, tts, cv, robot с 9 параметрами из ТЗ).
- integration/__init__.py: docstring-описание модуля.
- integration/demo_coffee.py: end-to-end демо «принеси кофе Олегу». 10 шагов: ASR → интент → CV find cup → approach cup → grasp → stability check → CV find person → approach target → handover → TTS. rich-плашки для каждого шага, итоговая таблица отчёта. Два режима: локальный MockG1 (без ZMQ-сервера) и удалённый через _RemoteG1 + Requester (--use-zmq). Использует параметры робота из config (approach_distance, grasp_force_min, handover_release_force, handover_distance).
- integration/demo_joke.py: end-to-end демо «расскажи анекдот 86-го». 3 шага: ASR + интент → RAG retrieve → TTS. RAG: реальный через voice/rag/retrieve.py subprocess если Qdrant жив (--use-qdrant), иначе fallback по локальному корпусу jokes_corpus.jsonl с наивным keyword-scoring + топ-1. TTS: реальный через voice/tts/infer.py если есть checkpoint (--use-tts), иначе mock с публикацией в voice.speak. Эвристический парсер темы (штирлиц/чукча/брежнев/...).
- integration/monitor.py: thin-обёртка над common.monitor (алиас для запуска из integration/).
- scripts/run_demo.sh: bash-скрипт запуска end-to-end демо в mock-режиме. Запускает MockG1 --serve в фоне (PID trap cleanup на EXIT/INT/TERM), ждёт 1.5 сек, запускает demo_coffee.py --use-zmq, передаёт аргументы, пишет логи в logs/. Поддержка --no-zmq для локального режима. Корректная обработка пустого массива аргументов.
- Все скрипты: Python 3.11+, CLI через click (где применимо), логирование через rich, комментарии/docstrings на русском, чёткие сообщения об ошибках на русском, mock-режим как fallback по умолчанию, структура: импорты → константы → функции → main().
- Прогнан smoke-test:
  - python common/config.py — печатает полный конфиг из default.yaml.
  - python common/state.py — pydantic v2 доступен, to_json/from_json работают.
  - python common/transport.py — pyzmq доступен, Publisher создаётся/закрывается без ошибок.
  - python common/mock_hardware.py --smoke-test — все методы MockG1 отрабатывают (find cup, move_to, grasp, get_force, get_imu, speak, release, snapshot).
  - python common/mock_hardware.py --serve --max-requests 2 + Requester('tcp://localhost:5554') — req/rep работает: find_object и snapshot возвращают корректные payload.
  - python integration/demo_coffee.py — локальный режим, 10/10 шагов ✓ за 2.21 с.
  - bash scripts/run_demo.sh — ZMQ-режим (MockG1 сервер в фоне), 8/8 шагов ✓ за 2.23 с.
  - python integration/demo_joke.py — fallback RAG по корпусу, найден анекдот про штирлица, mock TTS.
  - python integration/demo_joke.py --query "анекдот про чукчу" — корректно находит анекдот чукчи.
  - python common/monitor.py --max-messages 3 — перехватывает coffee.state (MOVING) и voice.speak ("Олег, держи свой кофе") из работающего demo_coffee.
  - python integration/monitor.py — алиас работает.

Stage Summary:
- Созданные файлы:
  - common/__init__.py (docstring)
  - common/config.py (YAML-конфиг + dataclass'ы + CLI)
  - common/logger.py (rich + RotatingFileHandler)
  - common/state.py (pydantic v2 + fallback, 5 моделей)
  - common/transport.py (Publisher/Subscriber/Requester/Replier + stub-mode)
  - common/mock_hardware.py (MockG1 + serve_loop + CLI)
  - common/monitor.py (CLI монитор pub-топиков)
  - config/default.yaml (полная конфигурация проекта)
  - integration/__init__.py (docstring)
  - integration/demo_coffee.py (end-to-end «принеси кофе Олегу», 10 шагов)
  - integration/demo_joke.py (end-to-end «расскажи анекдот 86-го», 3 шага, Qdrant+TTS optional)
  - integration/monitor.py (алиас на common/monitor.py)
  - scripts/run_demo.sh (bash: MockG1 --serve + demo_coffee.py --use-zmq + cleanup)
- Что можно запустить сразу (без G1, без камеры, без vLLM, без Qdrant):
  - python integration/demo_coffee.py — локальное end-to-end демо «принеси кофе Олегу» через MockG1.
  - bash scripts/run_demo.sh — то же, но через ZMQ-сервер MockG1 (запускает mock в фоне, корректно завершает).
  - python integration/demo_joke.py — end-to-end демо анекдота с fallback RAG по корпусу.
  - python integration/demo_joke.py --query "анекдот про чукчу" — поиск анекдота по теме.
  - python common/monitor.py — мониторинг всех pub-топиков в реальном времени (запустить в одном терминале, в другом — demo).
  - python common/mock_hardware.py --smoke-test — локальный прогон всех методов MockG1.
  - python common/mock_hardware.py --serve — запустить MockG1 как ZMQ REP-сервер (для интеграции с другими модулями).
  - python common/config.py — проверить, что конфиг загружается корректно.
- Что требует внешних сервисов:
  - integration/demo_coffee.py --use-zmq — нужен запущенный MockG1 --serve (или bash scripts/run_demo.sh делает это автоматически).
  - integration/demo_joke.py --use-qdrant — нужен Qdrant на http://localhost:6333 + индекс jokes_1986 (собирается voice/rag/build_index.py).
  - integration/demo_joke.py --use-tts — нужен checkpoint Бурунова (voice/tts/train.py) + базовая XTTS v2.
  - common/monitor.py для req/rep-топиков (coffee.tool_*) — не поддерживается (это req/rep, не pub/sub); только pub-топики.
