from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from triangulate import process_dem_file
from fastapi import Form

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    subset_size: int = Form(20)  # Використовуємо Form() замість просто int
):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        f.write(await file.read())

    points, triangles = process_dem_file(filepath, subset_size)
    
    return JSONResponse(content={
        "points": points.tolist() if hasattr(points, 'tolist') else points,
        "triangles": triangles.tolist() if hasattr(triangles, 'tolist') else triangles
    })

