from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from deepface import DeepFace
import numpy as np
import pickle
import os
import shutil
from datetime import datetime
import csv

app = FastAPI()

# --------- Setup Paths ---------
embedding_file = r"C:\MyDocuments\Attendance system Deepface\Backend\embeddings.pkl"
upload_dir = r"C:\MyDocuments\Attendance system Deepface\Backend\uploads"
attendance_dir = r"C:\MyDocuments\Attendance system Deepface\Backend\attendance_logs"
templates = Jinja2Templates(directory="templates")

os.makedirs(upload_dir, exist_ok=True)
os.makedirs(attendance_dir, exist_ok=True)

# --------- Load Embeddings ---------
with open(embedding_file, "rb") as f:
    saved_embeddings = pickle.load(f)

folder_embeddings = {}
for record in saved_embeddings:
    person = record["person"]
    embedding = record["embedding"]
    if person not in folder_embeddings:
        folder_embeddings[person] = []
    folder_embeddings[person].append(embedding)

# --------- Load Face Recognition Model ---------
model_name = 'Facenet'
detector_backend = 'opencv'
distance_threshold = 0.6
model = DeepFace.build_model(model_name)

# --------- Routes ---------

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("face_recognition.html", {"request": request})


@app.post("/recognize_face", response_class=HTMLResponse)
async def recognize_face(request: Request, file: UploadFile = File(...)):
    try:
        temp_image_path = os.path.join(upload_dir, f"{datetime.now().timestamp()}_{file.filename}")
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            input_faces = DeepFace.represent(
                img_path=temp_image_path,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False
            )
        except Exception as e:
            os.remove(temp_image_path)
            return templates.TemplateResponse("result.html", {
                "request": request,
                "status": "fail",
                "reason": f"Face detection error: {e}"
            })

        recognized_persons = []
        already_marked_today = set()
        today_csv_path = os.path.join(attendance_dir, datetime.now().strftime("%Y-%m-%d") + "_attendance.csv")

        # Load already marked people today
        if os.path.exists(today_csv_path):
            with open(today_csv_path, "r") as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)
                for row in reader:
                    already_marked_today.add(row[0])

        max_embeddings = max(len(emb_list) for emb_list in folder_embeddings.values())

        for face in input_faces:
            input_embedding = face["embedding"]
            found_person = None
            min_distance = float('inf')

            for i in range(max_embeddings):
                for person, embeddings in folder_embeddings.items():
                    if i >= len(embeddings):
                        continue

                    db_embedding = embeddings[i]
                    cosine_distance = np.dot(input_embedding, db_embedding) / (
                        np.linalg.norm(input_embedding) * np.linalg.norm(db_embedding)
                    )
                    cosine_distance = 1 - cosine_distance

                    print(f"Comparing with: {person} (Image {i+1}), Distance: {cosine_distance:.4f}")

                    if cosine_distance < distance_threshold:
                        found_person = person
                        min_distance = cosine_distance
                        break

                if found_person:
                    break

            if found_person and found_person not in recognized_persons:
                recognized_persons.append(found_person)

        os.remove(temp_image_path)

        # Save attendance to CSV
        if recognized_persons:
            with open(today_csv_path, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                if os.stat(today_csv_path).st_size == 0:
                    writer.writerow(["Person", "Timestamp"])
                for person in recognized_persons:
                    if person not in already_marked_today:
                        writer.writerow([person, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

            return templates.TemplateResponse("result.html", {
                "request": request,
                "status": "success",
                "recognized_persons": recognized_persons
            })

        else:
            return templates.TemplateResponse("result.html", {
                "request": request,
                "status": "fail",
                "reason": "No matching faces found"
            })

    except Exception as e:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "status": "error",
            "reason": str(e)
        })


@app.get("/dashboard", response_class=HTMLResponse)
async def show_dashboard(request: Request):
    today_csv_path = os.path.join(attendance_dir, datetime.now().strftime("%Y-%m-%d") + "_attendance.csv")
    records = []

    if os.path.exists(today_csv_path):
        with open(today_csv_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    records.append((row[0], row[1]))

    return templates.TemplateResponse("attendance_dashboard.html", {
        "request": request,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "records": records
    })
