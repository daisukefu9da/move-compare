#!/bin/bash
# Mac 用: ダブルクリックで最新版を取り込んで画面を起動する
cd "$(dirname "$0")"
PY=$(command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3)
if [ -z "$PY" ]; then
  echo "Python が見つかりません。https://www.python.org/ から Python 3.11 以上を入れてください"
  read -r -p "Enter で閉じます"
  exit 1
fi
"$PY" launcher.py
read -r -p "終了しました。Enter で閉じます"
