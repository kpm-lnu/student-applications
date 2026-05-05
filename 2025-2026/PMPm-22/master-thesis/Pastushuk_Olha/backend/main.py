import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference import EmotionPredictor 

# 1. Ініціалізація додатка
app = FastAPI(title="EmotionCRNN VAD Analyzer API")

# 2. Налаштування CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Шлях до моделі
CHECKPOINT_PATH = "checkpoints/emotion_crnn_best.pth"

# Завантажуємо Predictor при старті
if os.path.exists(CHECKPOINT_PATH):
    predictor = EmotionPredictor(CHECKPOINT_PATH)
    print(f"--- Модель успішно завантажена з {CHECKPOINT_PATH} ---")
else:
    print(f"--- ПОПЕРЕДЖЕННЯ: Чекпоїнт {CHECKPOINT_PATH} не знайдено! ---")
    predictor = None

TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Emotion Analysis API is running"}

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """Ендпоінт для отримання аудіо та повернення VAD результатів з урахуванням VAD перевірки"""
    
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server")

    file_extension = file.filename.split(".")[-1]
    temp_filename = f"{uuid.uuid4()}.{file_extension}"
    temp_path = os.path.join(TEMP_DIR, temp_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        results = predictor.predict(temp_path)
        results["filename"] = file.filename
        return results

    except Exception as e:
        print(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)