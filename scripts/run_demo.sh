#!/bin/bash
# scripts/run_demo.sh — запуск end-to-end демо в mock-режиме.
#
# Что делает:
#   1. Запускает MockG1 в фоне (python common/mock_hardware.py --serve).
#   2. Запускает integration/demo_coffee.py с --use-zmq.
#   3. По завершении — корректно убивает mock-процесс.
#
# Запуск:
#   bash scripts/run_demo.sh
#   bash scripts/run_demo.sh --command "принеси кофе Олегу"
#   bash scripts/run_demo.sh --use-zmq   # явный флаг (по умолчанию уже включён)
#
# Не требует реального G1, камер, vLLM, Qdrant — всё mock.

set -euo pipefail

# --- Константы ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python3}"
MOCK_LOG="$PROJECT_ROOT/logs/mock_g1.log"
DEMO_LOG="$PROJECT_ROOT/logs/demo_coffee.log"

# Переходим в корень проекта (чтобы относительные пути в конфиге работали)
cd "$PROJECT_ROOT"

mkdir -p "$PROJECT_ROOT/logs"

# Парсим аргументы командной строки: всё, что передано, отдаём demo_coffee.py
DEMO_ARGS=("$@")
# По умолчанию — используем ZMQ-сервер MockG1
USE_ZMQ=1
for arg in "${DEMO_ARGS[@]:-}"; do
  if [[ "$arg" == "--no-zmq" ]]; then
    USE_ZMQ=0
  fi
done

echo "============================================================"
echo " G1 EDU — end-to-end demo (mock-режим)"
echo " Корень проекта: $PROJECT_ROOT"
echo " Python: $PYTHON"
echo " Use ZMQ: $USE_ZMQ"
echo " Логи:    $PROJECT_ROOT/logs/"
echo "============================================================"

MOCK_PID=""
cleanup() {
  if [[ -n "$MOCK_PID" ]] && kill -0 "$MOCK_PID" 2>/dev/null; then
    echo ""
    echo "[run_demo] Останавливаю MockG1 (PID $MOCK_PID)..."
    kill "$MOCK_PID" 2>/dev/null || true
    wait "$MOCK_PID" 2>/dev/null || true
    echo "[run_demo] MockG1 остановлен."
  fi
}
trap cleanup EXIT INT TERM

# --- 1. Запуск MockG1 в фоне (если USE_ZMQ=1) ---
if [[ "$USE_ZMQ" -eq 1 ]]; then
  echo "[run_demo] Запуск MockG1 как ZMQ-сервера (порт 5554)..."
  $PYTHON "$PROJECT_ROOT/common/mock_hardware.py" --serve \
    >"$MOCK_LOG" 2>&1 &
  MOCK_PID=$!
  echo "[run_demo] MockG1 PID=$MOCK_PID, лог: $MOCK_LOG"
  # Даём серверу стартовать (bind ZMQ-сокета)
  sleep 1.5
  if ! kill -0 "$MOCK_PID" 2>/dev/null; then
    echo "[run_demo][ОШИБКА] MockG1 упал при старте. См. лог: $MOCK_LOG"
    cat "$MOCK_LOG" || true
    exit 1
  fi
else
  echo "[run_demo] --no-zmq: MockG1 запускается локально внутри demo_coffee.py"
fi

# --- 2. Запуск demo_coffee.py ---
echo ""
echo "[run_demo] Запуск demo_coffee.py с аргументами: ${DEMO_ARGS[*]:-}"
echo ""

if [[ "$USE_ZMQ" -eq 1 ]]; then
  # Убираем --no-zmq если он был, добавляем --use-zmq
  ARGS=()
  for a in "${DEMO_ARGS[@]:-}"; do
    [[ -z "$a" ]] && continue
    [[ "$a" == "--no-zmq" ]] && continue
    ARGS+=("$a")
  done
  if [[ ${#ARGS[@]} -gt 0 ]]; then
    $PYTHON "$PROJECT_ROOT/integration/demo_coffee.py" \
      --use-zmq "${ARGS[@]}" 2>&1 | tee "$DEMO_LOG"
  else
    $PYTHON "$PROJECT_ROOT/integration/demo_coffee.py" \
      --use-zmq 2>&1 | tee "$DEMO_LOG"
  fi
else
  if [[ ${#DEMO_ARGS[@]} -gt 0 ]]; then
    $PYTHON "$PROJECT_ROOT/integration/demo_coffee.py" \
      "${DEMO_ARGS[@]}" 2>&1 | tee "$DEMO_LOG"
  else
    $PYTHON "$PROJECT_ROOT/integration/demo_coffee.py" \
      2>&1 | tee "$DEMO_LOG"
  fi
fi

DEMO_RC=${PIPESTATUS[0]}
echo ""
echo "[run_demo] demo_coffee.py завершился с кодом $DEMO_RC"

# --- 3. cleanup выполнится через trap ---
exit "$DEMO_RC"
