# Voice — TTS (клон голоса С. Бурунова)

Обучение и инференс Text-to-Speech для клонирования голоса Сергея Бурунова.

## Стек
- **Базовая модель:** XTTS v2 (Coqui) — рекомендуется для русского, либо F5-TTS как альтернатива
- **Датасет:** аудио/видео с голосом С. Бурунова
- **Инструменты:** Whisper (распознавание речи для выравнивания), silero-vad (нарезка по тишине), torchaudio

## Установка
```bash
cd voice/tts
pip install -r requirements.txt
```

## Датасет
Складывать в `voice/data/voice_samples/`:
- `raw/` — исходные аудио/видео (mp3, wav, mp4)
- `processed/` — нарезанные фразы 3–15 сек, 22050 Гц, моно

## Обучение
```bash
python train.py \
  --dataset ../data/voice_samples/processed \
  --base_model xtts_v2 \
  --epochs 10 \
  --output ./checkpoints
```

## Инференс (генерация речи)
```bash
python infer.py \
  --text "Олег, держи свой кофе, бля" \
  --checkpoint ./checkpoints/best.pt \
  --out ./out.wav
```

## Задачи
- [ ] Собрать датасет голоса Бурунова (минимум 10 минут чистого голоса)
- [ ] Написать скрипт предобработки (`preprocess.py`)
- [ ] Тестовое обучение и слепая оценка качества (MOS)
- [ ] Оптимизация под Jetson Orin (ONNX / TensorRT)
