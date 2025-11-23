@echo off
cd /d C:\Users\olive\OneDrive\Desktop\LexiMind\LexiMind
call C:\Users\olive\OneDrive\Desktop\LexiMind\.venv\Scripts\activate.bat
python scripts\train.py --training-config configs\training\default.yaml --model-config configs\model\base.yaml --data-config configs\data\datasets.yaml --device cuda > logs\training_live.log 2>&1
