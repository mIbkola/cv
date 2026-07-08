# Зависимости и контакты

## Источники
- Unitree G1 EDU SDK: https://github.com/unitreerobotics/unitree_sdk2
- Unitree G1 Python SDK: https://github.com/unitreerobotics/unitree_sdk2_python
- Livox MID-360 driver: https://github.com/Livox-SDK/livox_ros_driver2
- Intel RealSense: https://github.com/IntelRealSense/librealsense
- Coqui XTTS: https://github.com/coqui-ai/TTS
- Ultralytics YOLO: https://github.com/ultralytics/ultralytics

## Железо
- Робот: Unitree G1 EDU Ultimate D
- Камера глубины: Intel RealSense D435
- 3D Lidar: Livox MID-360
- Захваты: RH56DFTP × 2 (не родные — требуется калибровка)
- Опционально: NVIDIA Jetson Orin для on-board compute

## TODO
- [ ] Подтвердить доступ к железу
- [ ] Получить SDK и драйвера
- [ ] Настроить окружение на Jetson Orin (Ubuntu 22.04 + CUDA)
- [ ] Собрать датасеты (голос Бурунова, анекдоты 1986, фото стаканов)
