import os
import shutil
import csv
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from deepface import DeepFace

from ..core.config import supabase, UPLOAD_DIR, ATTENDANCE_DIR, templates
from ..core.location import validate_user_location
from Backend.UI.core.security import get_session

router = APIRouter()

# ================= DEEPFACE CONFIG =================
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"
DISTANCE_THRESHOLD = 0.6

DeepFace.build_model(MODEL_NAME)


@router.post("/recognize_face", response_class=HTMLResponse)
async def recognize_face(
    request: Request,
    file: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
):
    session = get_session(request)

    if not session or not session.get("logged_in"):
        return RedirectResponse("/login", status_code=302)

    role = session["role"]
    logged_in_user_id = session["user_id"]
    back_url = "/admin" if role == "admin" else "/user"

    # ---------- LOCATION CHECK ----------
    if latitude is None or longitude is None:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "status": "fail",
                "reason": "Location permission required",
                "back_url": back_url
            }
        )

    is_allowed, reason = validate_user_location(latitude, longitude)
    if not is_allowed:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "status": "fail",
                "reason": reason,
                "back_url": back_url
            }
        )

    # ---------- SAVE IMAGE ----------
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    img_path = os.path.join(
        UPLOAD_DIR,
        f"{int(datetime.now().timestamp())}_{file.filename}"
    )

    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ---------- FACE EMBEDDING ----------
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
            {
                "request": request,
                "status": "error",
                "reason": str(e),
                "back_url": back_url
            }
        )

    recognized_names = []

    # ---------- FACE MATCHING ----------
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

        if role == "user" and matched_user_id != logged_in_user_id:
            os.remove(img_path)
            return templates.TemplateResponse(
                "result.html",
                {
                    "request": request,
                    "status": "fail",
                    "reason": "You can mark attendance only for yourself",
                    "back_url": back_url
                }
            )

        recognized_names.append(matched_name)

    os.remove(img_path)

    if not recognized_names:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "status": "fail",
                "reason": "No matching face found",
                "back_url": back_url
            }
        )

    # ---------- WRITE ATTENDANCE ----------
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    today_csv = os.path.join(
        ATTENDANCE_DIR,
        f"{datetime.now().strftime('%Y-%m-%d')}_attendance.csv"
    )

    file_exists = os.path.exists(today_csv)

    with open(today_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Person", "Timestamp"])
        for name in set(recognized_names):
            writer.writerow([name, datetime.now().strftime("%H:%M:%S")])

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "status": "success",
            "recognized_persons": list(set(recognized_names)),
            "back_url": back_url
        }
    )
