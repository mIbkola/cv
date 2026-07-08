# Voice — TTS (клон голоса С. Бурунова)

Обучение и инференс Text-to-Speech для клонирования голоса Сергея Бурунова.

## Стек
- **Базовая модель:** XTTS v2 (Coqui) — рекомендуется для русского, либо F5-TTS как альтернатива
- **Датасет:** аудио/видео с голосом С. Бурунова
- **Инструменты:** Whisper (ASR для выравнивания), silero-vad (VAD), torchaudio

## Установка
```bash
cd voice/tts
pip install -r requirements.txt
```

## Датасет
Сложить в `voice/data/voice_samples/`:
- `raw/` — исходные аудио/видео (mp3, wav, mp4)
- `processed/` — нарезанные фразы 3–15 сек, 22050 Hz, моно

## Обучение
```bash
python train.py \
  --dataset ../data/voice_samples/processed \
  --base_model xtts_v2 \
  --epochs 10 \
  --output ./checkpoints
```

## Инференс
```bash
python infer.py \
  --text "Олег, держи свой кофе, бля" \
  --checkpoint ./checkpoints/best.pt \
  --out ./out.wav
```

## TODO
- [ ] Собрать датасет голоса Бурунова (≥ 10 мин чистого)
- [ ] Скрипт предобработки (`preprocess.py`)
- [ ] Тестовое обучение и слепая оценка MOS
- [ ] Оптимизация под Jetson Orin (ONNX / TensorRT)
