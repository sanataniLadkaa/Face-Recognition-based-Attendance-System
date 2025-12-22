import os
import time
import cv2
from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from deepface import DeepFace
from supabase import create_client
from dotenv import load_dotenv

# ---------------- ENV ----------------
load_dotenv(r"C:\MyDocuments\Attendance system Deepface\.env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- ROUTER ----------------
router = APIRouter()

# ---------------- PATHS ----------------
BASE = r"C:\MyDocuments\Attendance system Deepface\Backend"
TEMP_DIR = os.path.join(BASE, "temp_capture")
os.makedirs(TEMP_DIR, exist_ok=True)

# ---------------- MODEL ----------------
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"
DeepFace.build_model(MODEL_NAME)

# ---------------- RECORD UI ----------------

@router.get("/record", response_class=HTMLResponse)
async def record_ui():
    return HTMLResponse("""
    <h2>📷 Record Person</h2>
    <form action="/attendance/start-recording" method="post">
        <input name="label_name" placeholder="Person Name" required />
        <button type="submit">Start Recording</button>
    </form>
    <br>
    <a href="/attendance/manage">✏️ Manage Persons</a>
    """)

# ---------------- RECORD & UPLOAD ----------------
@router.post("/start-recording")
async def start_recording(label_name: str = Form(...)):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {"error": "Camera not accessible"}

    frame = 0
    start = time.time()
    inserted = 0

    os.makedirs(TEMP_DIR, exist_ok=True)

    try:
        while time.time() - start < 5:
            ret, img = cap.read()
            if not ret:
                break

            img_path = os.path.join(TEMP_DIR, f"{label_name}_{frame}.jpg")
            cv2.imwrite(img_path, img)

            try:
                rep = DeepFace.represent(
                    img_path=img_path,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False
                )

                embedding = list(map(float, rep[0]["embedding"]))

                # 🔥 Insert into Supabase
                supabase.table("face_embeddings").insert({
                    "person_name": label_name,
                    "embedding": embedding
                }).execute()

                inserted += 1

            except Exception as e:
                print(f"⚠️ Face processing failed: {e}")

            finally:
                # ✅ Always cleanup temp image
                if os.path.exists(img_path):
                    os.remove(img_path)

            frame += 1
            time.sleep(0.5)

    finally:
        cap.release()

    return {
        "saved_images": frame,
        "new_embeddings_created": inserted
    }

# ---------------- MANAGE PERSONS ----------------

@router.get("/manage", response_class=HTMLResponse)
async def manage_persons():
    result = supabase.table("face_embeddings") \
        .select("person_name") \
        .execute()

    persons = sorted(set(r["person_name"] for r in result.data))

    rows = "".join(f"""
    <tr>
        <td>{p}</td>
        <td>
            <form action="/attendance/rename" method="post" style="display:inline">
                <input type="hidden" name="old_name" value="{p}">
                <input name="new_name" placeholder="New name" required>
                <button>✏️ Rename</button>
            </form>
            <form action="/attendance/delete" method="post" style="display:inline">
                <input type="hidden" name="person" value="{p}">
                <button style="color:red">🗑️ Delete</button>
            </form>
        </td>
    </tr>
    """ for p in persons)

    return HTMLResponse(f"""
    <h2>✏️ Manage Persons</h2>
    <table border="1" cellpadding="10">
        <tr><th>Name</th><th>Actions</th></tr>
        {rows}
    </table>
    <br><a href="/home">⬅ Back</a>
    """)

# ---------------- RENAME ----------------

@router.post("/rename")
async def rename_person(old_name: str = Form(...), new_name: str = Form(...)):
    supabase.table("face_embeddings") \
        .update({"person_name": new_name}) \
        .eq("person_name", old_name) \
        .execute()

    return RedirectResponse("/attendance/manage", status_code=303)

# ---------------- DELETE ----------------

@router.post("/delete")
async def delete_person(person: str = Form(...)):
    supabase.table("face_embeddings") \
        .delete() \
        .eq("person_name", person) \
        .execute()

    return RedirectResponse("/attendance/manage", status_code=303)
