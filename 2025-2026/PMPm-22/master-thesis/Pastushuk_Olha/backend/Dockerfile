# Використовуємо легкий образ Python
FROM python:3.12-slim

# Встановлюємо системні залежності для аудіо та librosa
RUN apt-get update && apt-get install -y \
    libasound2-dev \
    portaudio19-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копіюємо залежності та встановлюємо їх
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо код бекенду та ваги моделі
COPY . .

# Відкриваємо порт для FastAPI
EXPOSE 8000

# Запуск сервера
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]