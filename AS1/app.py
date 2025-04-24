from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import os
from PIL import Image
import numpy as np
from AS1.test import (  # Adjusted import to work within the AS1 folder
    load_model_from_file,
    extract_labels_from_dataset,
    detect_face,
    LabelEncoder
)

# FastAPI instance
app = FastAPI()

# Static files and templates
app.mount("/static", StaticFiles(directory=os.path.join("AS1", "static")), name="static")
templates = Jinja2Templates(directory=os.path.join("AS1", "templates"))

# Constants
model_path = "face_recognition_model.h5"  # Adjusted path
dataset_path = os.path.join("AS1", "dataset")  # Adjusted path
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
    Render the main page where users can upload an image.
    """
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/predict", response_class=HTMLResponse)
async def predict_face(request: Request, file: UploadFile = File(...)):
    """
    Handle face recognition prediction for uploaded images.
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error": "Uploaded file is not an image."
        })
    
    # Save the uploaded file temporarily
    temp_dir = os.path.join("AS1", "uploads")  # Adjusted path for temporary uploads
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error": f"Failed to save the uploaded file: {e}"
        })

    # Detect and predict
    try:
        face = detect_face(file_path)
        if face is None:
            return templates.TemplateResponse("result.html", {
                "request": request,
                "error": "No face detected in the image."
            })
        
        # Convert face to PIL Image if necessary
        if isinstance(face, np.ndarray):
            face = Image.fromarray(face)

        # Preprocess the image and predict
        face_preprocessed = preprocess_image(face, target_size=image_size)
        prediction = model.predict(face_preprocessed)
        predicted_class_index = np.argmax(prediction, axis=1)
        predicted_class_name = label_encoder.inverse_transform(predicted_class_index)[0]

        # Move the uploaded file to static directory for display
        static_dir = os.path.join("AS1", "static", "uploads")
        os.makedirs(static_dir, exist_ok=True)

        # Generate a unique filename if the file already exists
        static_image_path = os.path.join(static_dir, file.filename)
        if os.path.exists(static_image_path):
            base, ext = os.path.splitext(file.filename)
            unique_filename = f"{base}_{uuid.uuid4().hex[:8]}{ext}"
            static_image_path = os.path.join(static_dir, unique_filename)

        os.rename(file_path, static_image_path)

        # Return the prediction result
        return templates.TemplateResponse("result.html", {
            "request": request,
            "predicted_class": predicted_class_name,
            "image_path": f"/static/uploads/{os.path.basename(static_image_path)}"  # Pass the image path for display
        })
    except Exception as e:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error": f"Error during face detection or prediction: {e}"
        })


def preprocess_image(face, target_size=(224, 224)):
    """
    Preprocess the image by resizing and normalizing.
    """
    # Ensure face is a PIL image and make it mutable
    if not isinstance(face, Image.Image):
        face = Image.open(face)  # Load face as a PIL Image if it's not already

    face_resized = face.copy()  # Make a copy to ensure mutability
    face_resized = face_resized.resize(target_size)  # Resize to the specified size
    face_array = np.array(face_resized)  # Convert image to numpy array
    face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension
    face_array = face_array / 255.0  # Normalize the pixel values to [0, 1]
    return face_array
