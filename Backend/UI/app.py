from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
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
    allow_origins=["*"],
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

# -------------------- HELPERS --------------------

def get_username_from_user_id(user_id: str) -> str:
    res = (
        supabase
        .table("face_embeddings")
        .select("person_name")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    return res.data[0]["person_name"] if res.data else "User"


def get_session(request: Request):
    return {
        "logged_in": request.cookies.get("logged_in") == "yes",
        "role": request.cookies.get("role"),
        "username": request.cookies.get("username"),
        "user_id": request.cookies.get("user_id"),
    }


def require_login(request: Request):
    if not get_session(request)["logged_in"]:
        return RedirectResponse("/login")


def require_admin(request: Request):
    s = get_session(request)
    if not s["logged_in"] or s["role"] != "admin":
        return RedirectResponse("/login")

# -------------------- ROOT --------------------

@app.get("/", response_class=HTMLResponse)
async def entry(request: Request):
    s = get_session(request)
    if not s["logged_in"]:
        return RedirectResponse("/login")
    return RedirectResponse("/admin" if s["role"] == "admin" else "/user")

# -------------------- LOGIN --------------------

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),  # user_id
    password: str = Form(...)
):
    role = verify_login(username, password)
    if not role:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid credentials"}
        )

    person_name = get_username_from_user_id(username)

    response = RedirectResponse(
        "/admin" if role == "admin" else "/user",
        status_code=302
    )

    response.set_cookie("logged_in", "yes", httponly=True)
    response.set_cookie("role", role, httponly=True)
    response.set_cookie("username", person_name, httponly=True)
    response.set_cookie("user_id", username, httponly=True)

    return response


@app.get("/logout")
async def logout():
    response = RedirectResponse("/login")
    for k in ["logged_in", "role", "username", "user_id"]:
        response.delete_cookie(k)
    return response

# -------------------- ADMIN UI --------------------

@app.get("/admin", response_class=HTMLResponse)
async def admin_home(request: Request):
    resp = require_admin(request)
    if resp:
        return resp
    return templates.TemplateResponse("face_recognition.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    resp = require_admin(request)
    if resp:
        return resp

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

# -------------------- USER UI --------------------

@app.get("/user", response_class=HTMLResponse)
async def user_dashboard(request: Request):
    s = get_session(request)
    if not s["logged_in"] or s["role"] != "user":
        return RedirectResponse("/login")

    return templates.TemplateResponse(
        "user_dashboard.html",
        {"request": request, "username": s["username"]}
    )

# -------------------- FACE RECOGNITION --------------------

@app.post("/recognize_face", response_class=HTMLResponse)
async def recognize_face(request: Request, file: UploadFile = File(...)):
    s = get_session(request)
    if not s["logged_in"]:
        return RedirectResponse("/login")

    role = s["role"]
    logged_in_user_id = s["user_id"]

    img_path = os.path.join(
        UPLOAD_DIR,
        f"{datetime.now().timestamp()}_{file.filename}"
    )

    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    faces = DeepFace.represent(
        img_path=img_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False
    )

    recognized = []

    for face in faces:
        embedding = list(map(float, face["embedding"]))

        result = supabase.rpc(
            "match_face_embedding",
            {
                "query_embedding": embedding,
                "match_threshold": DISTANCE_THRESHOLD,
                "match_count": 1
            }
        ).execute()

        if not result.data:
            continue

        matched_user_id = result.data[0]["user_id"]
        matched_name = result.data[0]["person_name"]

        # ❌ Restriction ONLY for normal users
        if role == "user" and matched_user_id != logged_in_user_id:
            os.remove(img_path)
            return templates.TemplateResponse(
                "result.html",
                {
                    "request": request,
                    "status": "fail",
                    "error": "You can mark attendance only for yourself"
                }
            )

        recognized.append(matched_name)

    os.remove(img_path)

    if not recognized:
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "status": "fail"}
        )

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

# -------------------- USER ATTENDANCE API --------------------

@app.get("/attendance/my")
async def my_attendance(request: Request):
    s = get_session(request)
    if not s["logged_in"] or s["role"] != "user":
        return JSONResponse(status_code=403, content={"error": "Unauthorized"})

    username = s["username"]
    records = []

    for file in os.listdir(ATTENDANCE_DIR):
        with open(os.path.join(ATTENDANCE_DIR, file), "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row[0] == username:
                    records.append({
                        "date": file[:10],
                        "time": row[1]
                    })

    return records

# -------------------- EXTRA ROUTER --------------------

app.include_router(attendance_router, prefix="/attendance")
