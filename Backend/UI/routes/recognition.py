import os
import shutil
import csv
from datetime import datetime

from fastapi import APIRouter, Request, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from deepface import DeepFace

from ..core.config import (
    supabase,
    UPLOAD_DIR,
    ATTENDANCE_DIR,
    templates
)
from UI.core.security import get_session

# ---------------- ROUTER ----------------
router = APIRouter()

# ---------------- DEEPFACE CONFIG ----------------
MODEL_NAME = "Facenet"          # ✅ STRING ONLY
DETECTOR_BACKEND = "opencv"
DISTANCE_THRESHOLD = 0.6

# ✅ Load ONCE at import time (DeepFace caches it)
DeepFace.build_model(MODEL_NAME)

# ---------------- FACE RECOGNITION ----------------
@router.post("/recognize_face", response_class=HTMLResponse)
async def recognize_face(
    request: Request,
    file: UploadFile = File(...)
):
    session = get_session(request)

    if not session["logged_in"]:
        return RedirectResponse("/login")

    role = session["role"]
    logged_in_user_id = session["user_id"]

    # ---------- Save uploaded image ----------
    img_path = os.path.join(
        UPLOAD_DIR,
        f"{datetime.now().timestamp()}_{file.filename}"
    )

    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ---------- Extract embeddings ----------
    try:
        faces = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,          # ✅ CORRECT
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False
        )
    except Exception as e:
        os.remove(img_path)
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "status": "error",
                "reason": str(e)
            }
        )

    recognized_names = []

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

        # ❌ Users can mark attendance ONLY for themselves
        if role == "user" and matched_user_id != logged_in_user_id:
            os.remove(img_path)
            return templates.TemplateResponse(
                "result.html",
                {
                    "request": request,
                    "status": "fail",
                    "reason": "You can mark attendance only for yourself"
                }
            )

        recognized_names.append(matched_name)

    os.remove(img_path)

    # ---------- No face matched ----------
    if not recognized_names:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "status": "fail",
                "reason": "No matching face found"
            }
        )

    # ---------- Write attendance ----------
    today_csv = os.path.join(
        ATTENDANCE_DIR,
        f"{datetime.now().strftime('%Y-%m-%d')}_attendance.csv"
    )

    with open(today_csv, "a", newline="") as f:
        writer = csv.writer(f)

        if os.stat(today_csv).st_size == 0:
            writer.writerow(["Person", "Timestamp"])

        for name in set(recognized_names):
            writer.writerow([name, datetime.now().strftime("%H:%M:%S")])

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "status": "success",
            "recognized_persons": list(set(recognized_names))
        }
    )
