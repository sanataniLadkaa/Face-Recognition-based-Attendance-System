from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import os
from PIL import Image
import numpy as np
import base64
from io import BytesIO
from AS1.test import (
    load_model_from_file,
    extract_labels_from_dataset,
    
    LabelEncoder
)
import cv2
import numpy as np

def detect_face(image):
    """
    Detect a face in the given image.

    :param image: Can be a file path, a numpy array, or a PIL Image.
    :return: Cropped face as a numpy array, or None if no face is detected.
    """
    # If the image is a file path, load it using cv2.imread
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise ValueError(f"Failed to load image from path: {image}")

    # If the image is a PIL Image, convert it to a numpy array
    elif isinstance(image, Image.Image):
        image = np.array(image)

    # Ensure the image is a valid numpy array
    if not isinstance(image, np.ndarray):
        raise ValueError("Invalid image format. Expected a file path, numpy array, or PIL Image.")

    # Convert to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load Haar cascade or another face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None  # No faces detected

    # Crop the first detected face
    x, y, w, h = faces[0]
    cropped_face = image[y:y + h, x:x + w]
    return cropped_face


def preprocess_image(face, target_size=(224, 224)):
    """
    Preprocess the image by resizing and normalizing.

    :param face: The face image (as a numpy array).
    :param target_size: The target size to resize the image to.
    :return: The preprocessed image in the required format.
    """
    # If the face is a numpy array, convert it to a PIL Image
    if not isinstance(face, Image.Image):
        face = Image.fromarray(face)

    # Resize the image to the target size (e.g., 224x224 for models like VGG16, ResNet)
    face_resized = face.resize(target_size)

    # Convert the image to a numpy array and normalize it (scale pixel values between 0 and 1)
    face_array = np.array(face_resized) / 255.0

    # Expand the dimensions to match the expected input shape for the model (e.g., (1, height, width, channels))
    face_array = np.expand_dims(face_array, axis=0)

    return face_array


# FastAPI instance
app = FastAPI()

# Static files and templates
app.mount("/static", StaticFiles(directory=os.path.join("AS1", "static")), name="static")
templates = Jinja2Templates(directory=os.path.join("AS1", "templates"))

# Constants
model_path = "face_recognition_model.h5"
dataset_path = os.path.join("AS1", "dataset")
image_size = (224, 224)

# Load the model and prepare the label encoder
try:
    labels = extract_labels_from_dataset(dataset_path)
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    model = load_model_from_file(model_path)
except FileNotFoundError as e:
    raise HTTPException(status_code=500, detail=f"Model or dataset not found: {e}")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading the model: {e}")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Render the camera page for capturing an image.
    """
    return templates.TemplateResponse("camera.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict_face(request: Request, image_data: str = Form(...)):
    """
    Handle face recognition prediction for captured images.
    """
    try:
        # Decode the base64 image
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_bytes))

        # Detect and preprocess the face
        face = detect_face(image)  # Pass PIL Image
        if face is None:
            return templates.TemplateResponse("result.html", {
                "request": request,
                "error": "No face detected in the captured image."
            })

        # Save the detected face to a static folder
        face_image_path = os.path.join("AS1", "static", "detected_face.jpg")
        face_pil = Image.fromarray(face)
        face_pil.save(face_image_path)

        # Preprocess and predict
        face_preprocessed = preprocess_image(face, target_size=image_size)
        prediction = model.predict(face_preprocessed)
        predicted_class_index = np.argmax(prediction, axis=1)
        predicted_class_name = label_encoder.inverse_transform(predicted_class_index)[0]

        # Return the prediction result with the image path
        return templates.TemplateResponse("result.html", {
            "request": request,
            "predicted_class": predicted_class_name,
            "image_path": "/static/detected_face.jpg"
        })
    except Exception as e:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error": f"Error during prediction: {e}"
        })
