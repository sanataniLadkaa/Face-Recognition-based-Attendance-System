from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from login import verify_login
from store import router as attendance_router
from deepface import DeepFace
from supabase import create_client
from dotenv import load_dotenv

import os
import shutil
import csv
from datetime import datetime

# -------------------- ENV --------------------

load_dotenv(r"C:\MyDocuments\Attendance system Deepface\.env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("❌ Supabase env variables not loaded")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------- APP --------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- PATHS --------------------

BASE = r"C:\MyDocuments\Attendance system Deepface\Backend"

UPLOAD_DIR = os.path.join(BASE, "uploads")
ATTENDANCE_DIR = os.path.join(BASE, "attendance_logs")
TEMPLATES_DIR = os.path.join(BASE, "UI", "templates")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# -------------------- DEEPFACE --------------------

MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"
DISTANCE_THRESHOLD = 0.6

DeepFace.build_model(MODEL_NAME)

# -------------------- SESSION --------------------

def is_logged_in(request: Request):
    return request.cookies.get("logged_in") == "yes"

# -------------------- ROUTES --------------------

@app.get("/", response_class=HTMLResponse)
async def entry(request: Request):
    return RedirectResponse("/home" if is_logged_in(request) else "/login")

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):
    if verify_login(username, password):
        response = RedirectResponse("/home", status_code=302)
        response.set_cookie("logged_in", "yes", httponly=True)
        return response

    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Invalid credentials"}
    )

@app.get("/logout")
async def logout():
    response = RedirectResponse("/login")
    response.delete_cookie("logged_in")
    return response

@app.get("/home", response_class=HTMLResponse)
async def home_page(request: Request):
    if not is_logged_in(request):
        return RedirectResponse("/login")
    return templates.TemplateResponse("face_recognition.html", {"request": request})

# -------------------- FACE RECOGNITION --------------------

@app.post("/recognize_face", response_class=HTMLResponse)
async def recognize_face(request: Request, file: UploadFile = File(...)):
    if not is_logged_in(request):
        return RedirectResponse("/login")

    img_path = os.path.join(
        UPLOAD_DIR,
        f"{datetime.now().timestamp()}_{file.filename}"
    )

    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        faces = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False
        )
    except Exception as e:
        os.remove(img_path)
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "status": "error", "reason": str(e)}
        )

    recognized = []

    for face in faces:
        embedding = list(map(float, face["embedding"]))  # 🔥 IMPORTANT

        result = supabase.rpc(
            "match_face_embedding",
            {
                "query_embedding": embedding,
                "match_threshold": DISTANCE_THRESHOLD,
                "match_count": 1
            }
        ).execute()

        print("Supabase result:", result.data)

        if result.data:
            recognized.append(result.data[0]["person_name"])

    os.remove(img_path)

    if recognized:
        today_csv = os.path.join(
            ATTENDANCE_DIR,
            f"{datetime.now().strftime('%Y-%m-%d')}_attendance.csv"
        )

        with open(today_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if os.stat(today_csv).st_size == 0:
                writer.writerow(["Person", "Timestamp"])
            for p in set(recognized):
                writer.writerow([p, datetime.now().strftime("%H:%M:%S")])

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "status": "success",
                "recognized_persons": list(set(recognized))
            }
        )

    return templates.TemplateResponse(
        "result.html",
        {"request": request, "status": "fail", "reason": "No match found"}
    )

# -------------------- DASHBOARD --------------------

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not is_logged_in(request):
        return RedirectResponse("/login")

    today_csv = os.path.join(
        ATTENDANCE_DIR,
        f"{datetime.now().strftime('%Y-%m-%d')}_attendance.csv"
    )

    records = []
    if os.path.exists(today_csv):
        with open(today_csv, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            records.extend(reader)

    return templates.TemplateResponse(
        "attendance_dashboard.html",
        {
            "request": request,
            "records": records,
            "date": datetime.now().strftime("%Y-%m-%d"),
        }
    )

# -------------------- ROUTER --------------------
app.include_router(attendance_router, prefix="/attendance")
