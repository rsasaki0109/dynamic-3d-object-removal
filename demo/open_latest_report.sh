#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

CLEAN_MODE="safe"
if [[ "${1:-}" == "--force" ]]; then
  CLEAN_MODE="force"
elif [[ "${1:-}" == "--safe" ]]; then
  CLEAN_MODE="safe"
elif [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: ./open_latest_report.sh [--safe|--force]"
  echo "  --safe  : close only demo-related browser tabs/processes (default)"
  echo "  --force : also close broad index_3d*.html related processes"
  exit 0
elif [[ "${1:-}" != "" ]]; then
  echo "Unknown option: ${1:-}"
  echo "Usage: ./open_latest_report.sh [--safe|--force]"
  exit 1
fi

close_legacy_server() {
  # Stop previous demo server on 4173
  pkill -f "python3 -m http.server 4173" >/dev/null 2>&1 || true
  pkill -f "python3 -m http.server.*--bind .*4173" >/dev/null 2>&1 || true
  pkill -f "python3 -m http.server --directory $REPO_DIR 4173" >/dev/null 2>&1 || true
}

close_demo_browsers() {
  # Close tabs/windows that may still show an old report
  local browser_pattern=$1
  local -a base_patterns=(
    "127\\.0\\.0\\.1:4173/index(\\.html|_3d(_standalone)?\\.html)"
    "localhost:4173/index(\\.html|_3d(_standalone)?\\.html)"
    "file://.*/dynamic_object_removal/demo/index(\\.html|_3d(_standalone)?\\.html)"
    "file://.*/dynamic_object_removal/demo/index_3d\\.html"
    "demo/index(\\.html|_3d(_standalone)?\\.html)"
    "$REPO_DIR/index(\\.html|_3d(_standalone)?\\.html)"
  )
  local p
  for p in "${base_patterns[@]}"; do
    pkill -f "$p" >/dev/null 2>&1 || true
  done

  local -a browser_patterns
  if [[ "$browser_pattern" == "force" ]]; then
    browser_patterns=(
      "index(\\.html|_3d(_standalone)?\\.html)"
      "xdg-open .*index(\\.html|_3d(_standalone)?\\.html)"
    )
  else
    browser_patterns=(
      "xdg-open .*demo/index(\\.html|_3d(_standalone)?\\.html)"
      "xdg-open .*file://.*/dynamic_object_removal/demo/index(\\.html|_3d(_standalone)?\\.html)"
      "xdg-open .*$REPO_DIR/index(\\.html|_3d(_standalone)?\\.html)"
    )
  fi

  if [[ "$browser_pattern" != "force" ]]; then
    browser_patterns+=(
      "firefox .*(demo|$REPO_DIR)/index(\\.html|_3d(_standalone)?\\.html)"
      "chrome .*(demo|$REPO_DIR)/index(\\.html|_3d(_standalone)?\\.html)"
      "chromium .*(demo|$REPO_DIR)/index(\\.html|_3d(_standalone)?\\.html)"
      "brave-browser .*(demo|$REPO_DIR)/index(\\.html|_3d(_standalone)?\\.html)"
    )
  else
    browser_patterns+=(
      "firefox .*index(\\.html|_3d(_standalone)?\\.html)"
      "chrome .*index(\\.html|_3d(_standalone)?\\.html)"
      "chromium .*index(\\.html|_3d(_standalone)?\\.html)"
      "brave-browser .*index(\\.html|_3d(_standalone)?\\.html)"
    )
  fi

  for p in "${browser_patterns[@]}"; do
    pkill -f "$p" >/dev/null 2>&1 || true
  done
}

close_legacy_server
close_demo_browsers "$CLEAN_MODE"

python3 run_demo.py

nohup python3 -m http.server 4173 --bind 127.0.0.1 >/tmp/dynamic_object_removal_demo_server.log 2>&1 < /dev/null &
SERVER_PID=$!

sleep 0.5
if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
  echo "Failed to start local HTTP server. Check /tmp/dynamic_object_removal_demo_server.log"
  exit 1
fi

if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "http://127.0.0.1:4173/index_3d.html" >/dev/null 2>&1 || true
else
  echo "Open manually: http://127.0.0.1:4173/index_3d.html"
fi
