# 🚘 License Plate Recognition – Setup Guide

Цей проєкт реалізує автоматичне розпізнавання автомобільних номерних знаків із відео з використанням YOLOv8, PaddleOCR та PostgreSQL.

---

## ⚙️ 1. Встановлення середовища

### 🔁 Клонування репозиторію

```bash
git clone https://github.com/your-username/license-plate-recognition.git
cd license-plate-recognition


python -m venv .venv
source .venv/bin/activate        # для Linux/macOS
# або
.venv\Scripts\activate           # для Windows


pip install --upgrade pip
pip install -r requirements.txt



project_root/
│
├── .venv/                      # віртуальне середовище (ігнорується Git)
├── data/                       # датасет: train/valid/test
├── licence_plates/            # оброблені дані або вивід (опціонально)
│
├── .env                        # змінні середовища (для PostgreSQL)
├── .env.example                # приклад конфігурації .env
├── .gitignore                  # виключення для git
│
├── license_plate_detector.pt  # натренована модель YOLOv8
├── main.py                    # основний скрипт для аналізу відео
├── sqldb.py                   # логіка бази даних (створення, запис)
├── requirements.txt           # список Python-залежностей
└── README.md                  # ця інструкція


Загрузіть відео в відповідну папку.

Натренована модель license_plate_detector.pt повинна бути в корені проєкту


