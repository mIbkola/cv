# Voice — RAG (анекдоты 1986)

RAG-система: по запросу «расскажи анекдот из 86-го» / «анекдот про штирлица» извлекает релевантный анекдот из базы и подает в TTS (голос Бурунова).

## Стек
- **Эмбеддинги:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Векторная БД:** Qdrant (локально, Docker)
- **LLM (опционально):** Qwen2.5-7B-Instruct (локально) или GPT-4o-mini (API)
- **ASR:** Whisper (если вход голосовой)

## Установка
```bash
cd voice/rag
pip install -r requirements.txt

# Qdrant
docker run -d -p 6333:6333 qdrant/qdrant
```

## Структура данных
`voice/data/jokes_corpus.jsonl` — построчно:
```json
{"id": "j_0001", "year": 1986, "topic": "штирлиц", "text": "..."}
```

## Сборка индекса
```bash
python build_index.py \
  --corpus ../data/jokes_corpus.jsonl \
  --qdrant-url http://localhost:6333 \
  --collection jokes_1986
```

## Pipeline (полный цикл)
```bash
python pipeline.py --input "расскажи анекдот 86 года про штирлица"
```
Логика:
1. ASR (если voice input) → текст
2. LLM извлекает intent: `{action: "joke", topic: "штирлиц", year: 1986}`
3. RAG retrieve top-5
4. Cross-encoder rerank → топ-1
5. Опционально: LLM-адаптация (например, добавить «а вы слышали...»)
6. TTS (Бурунов) → аудио → speaker

## TODO
- [ ] Собрать корпус анекдотов 1986 ± 5 лет (≥ 500 шт.)
- [ ] Разметить topic-ами (штирлиц, василий иванович, чукча, лечащий врач, и т.д.)
- [ ] Запустить Qdrant + проиндексировать
- [ ] Подключить LLM для intent parsing
- [ ] Подключить TTS из `voice/tts`
