#!/usr/bin/env bash
# ブラウザ版（PoC）をこの環境で通しで確かめる: ビルド → ローカルで配信 → Playwright で操作
#   bash web/e2e/run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
python web/build.py
python web/e2e/make_sample.py
python -m http.server 8600 --directory web/_site > .cache/e2e/http.log 2>&1 &
SERVER=$!
trap 'kill $SERVER' EXIT
sleep 1
NODE_PATH="$(npm root -g)" node web/e2e/poc_flow.js http://localhost:8600/ .cache/e2e/two_people.webm
