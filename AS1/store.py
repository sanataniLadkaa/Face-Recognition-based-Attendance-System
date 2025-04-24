from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pymongo import MongoClient
from gridfs import GridFS
import cv2
import time

app = FastAPI()

# Set up MongoDB connection and GridFS
client = MongoClient("mongodb://localhost:27017/")
db = client["image_database"]
fs = GridFS(db)

# HTML template for the UI
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Image Capture for Attendance System</title>
</head>
<body>
    <h2>Capture Images for Attendance System</h2>
    <form action="/start-recording" method="post">
        <label for="label_name">Enter Person's Label Name:</label>
        <input type="text" id="label_name" name="label_name" required>
        <button type="submit">Start Recording</button>
    </form>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_form():
    return HTMLResponse(content=html_template)

@app.post("/start-recording")
async def start_recording(label_name: str = Form(...)):
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {"error": "Failed to open the camera. Please check your camera device."}

    fps = 10
    start_time = time.time()
    frame_count = 0
    max_time = 10  # seconds

    print(f"Recording for {max_time} seconds and storing images in MongoDB...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            if elapsed_time > max_time:
                break

            # Encode frame as JPEG
            _, buffer = cv2.imencode(".jpg", frame)

            # Store the frame in MongoDB GridFS
            image_id = fs.put(buffer.tobytes(), filename=f"{label_name}_{frame_count}.jpg", metadata={"label": label_name})
            print(f"Stored frame {frame_count} with ID: {image_id}")

            frame_count += 1
            time.sleep(1 / fps)

    finally:
        # Ensure the camera is released
        cap.release()

    return {"message": f"Saved {frame_count} frames for label '{label_name}' in MongoDB."}
