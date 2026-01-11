import os
import shutil
import csv
import math
from datetime import datetime

import geocoder
from fastapi import APIRouter, Request, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from deepface import DeepFace

from ..core.config import (
    supabase,
    UPLOAD_DIR,
    ATTENDANCE_DIR,
    templates
)
from Backend.UI.core.security import get_session

# ---------------- ROUTER ----------------
router = APIRouter()

# ---------------- DEEPFACE CONFIG ----------------
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"
DISTANCE_THRESHOLD = 0.6

# Load model ONCE
DeepFace.build_model(MODEL_NAME)

# ---------------- OFFICE LOCATION ----------------
OFFICE_LATITUDE = 28.6139     # CHANGE to office latitude
OFFICE_LONGITUDE = 77.2090   # CHANGE to office longitude
MAX_DISTANCE_METERS = 100000     # 10 meters

# ---------------- DISTANCE HELPER ----------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # meters

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

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

    # ---------- GET LOCATION VIA IP ----------
    g = geocoder.ip("me")

    if not g.ok or not g.latlng:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "status": "fail",
                "reason": "Unable to determine your location"
            }
        )

    device_lat, device_lon = g.latlng

    distance = haversine_distance(
        device_lat,
        device_lon,
        OFFICE_LATITUDE,
        OFFICE_LONGITUDE
    )

    if distance > MAX_DISTANCE_METERS:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "status": "fail",
                "reason": "You are not inside office premises"
            }
        )

    # ---------- SAVE IMAGE ----------
    img_path = os.path.join(
        UPLOAD_DIR,
        f"{datetime.now().timestamp()}_{file.filename}"
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

    if not recognized_names:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "status": "fail",
                "reason": "No matching face found"
            }
        )

    # ---------- WRITE ATTENDANCE ----------
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
