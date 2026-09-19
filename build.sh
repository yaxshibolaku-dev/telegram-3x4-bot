#!/usr/bin/env bash
set -e

echo "=== 1. Kutubxonalarni o'rnatish ==="
pip install -r requirements.txt

echo "=== 2. u2netp AI modelini oldindan yuklab olish ==="
python -c "import rembg; rembg.new_session('u2netp')" || true

echo "=== Build muvaffaqiyatli yakunlandi! ==="
