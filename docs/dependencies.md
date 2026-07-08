# Зависимости и контакты

## Источники (откуда брать SDK и драйвера)
- Unitree G1 EDU SDK (C++): https://github.com/unitreerobotics/unitree_sdk2
- Unitree G1 Python SDK: https://github.com/unitreerobotics/unitree_sdk2_python
- Livox MID-360 драйвер: https://github.com/Livox-SDK/livox_ros_driver2
- Intel RealSense: https://github.com/IntelRealSense/librealsense
- Coqui XTTS (TTS): https://github.com/coqui-ai/TTS
- Ultralytics YOLO (CV): https://github.com/ultralytics/ultralytics
- Qdrant (векторная БД): https://github.com/qdrant/qdrant
- vLLM (для локального LLM): https://github.com/vllm-project/vllm

## Железо
- Робот: Unitree G1 EDU Ultimate D
- Камера глубины: Intel RealSense D435
- 3D Lidar: Livox MID-360
- Захваты: RH56DFTP × 2 (не родные — требуется калибровка)
- Опционально: NVIDIA Jetson Orin для on-board вычислений

## Compute (архитектура)
- **On-board (Jetson Orin):** CV (детекция стакана и Олега), манипуляция, локомоция — критичное по времени
- **Внешний GPU-сервер:** TTS (клон Бурунова), LLM (Qwen2.5-7B-Instruct), RAG (Qdrant) — тяжёлое

## Транспорт
- Python 3.11+ + ZeroMQ (pub/sub + req/rep)
- ROS2 пока не используем (отложено до масштабирования)

## Задачи
- [ ] Подтвердить доступ к железу
- [ ] Получить SDK и драйвера
- [ ] Настроить окружение на Jetson Orin (Ubuntu 22.04 + CUDA)
- [ ] Настроить внешний GPU-сервер (Ubuntu + CUDA + Docker)
- [ ] Собрать датасеты (голос Бурунова, анекдоты 1986, фото стаканов, фото Олега)
