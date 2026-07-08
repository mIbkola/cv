# Coffee — Detection (CV)

Обнаружение бумажного стакана с кофе в кадре, выдача 3D-координат для манипулятора.

## Стек
- YOLOv8 / RT-DETR (ultralytics)
- OpenCV
- pyrealsense2 (RealSense D435 — RGB + Depth)

## Установка
```bash
cd coffee/detection
pip install -r requirements.txt
```

## Этапы

### 1. Baseline-тест на COCO YOLO
```bash
python baseline_test.py --images test_images/ --class-id 41  # COCO "cup"
```
Оценить mAP и визуально — отличает ли бумажный стакан от кружки.

### 2. Сбор датасета (если baseline плохой)
- 200–500 фото бумажных стаканов с кофе (разные бренды: Starbucks, кофейни, plain white)
- Разметка: `labelme` или `roboflow`, формат YOLO
- Аугментации: освещение, ракурс, наличие пара, разные фоны

### 3. Fine-tune
```bash
python finetune.py \
  --data dataset.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640
```

### 4. Инференс + 3D-локализация
```bash
python infer.py --camera realsense --model best.pt
```
На выходе: bounding box + 3D-координаты центра стакана в системе координат камеры → пересчёт в базовую систему робота через TF.

## TODO
- [ ] Собрать тестовый набор изображений бумажных стаканов
- [ ] Прогнать baseline YOLOv8 на COCO
- [ ] Принять решение: fine-tune или нет
- [ ] Реализовать 3D-локализацию через RealSense depth
