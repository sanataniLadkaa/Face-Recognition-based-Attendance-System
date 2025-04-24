import os
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import logging
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
image_size = (224, 224)
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Dynamic label extraction from dataset folder
def extract_labels_from_dataset(dataset_path):
    """
    Extract labels (person's names) from the folder names in the dataset directory.
    """
    labels = []
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            labels.append(folder_name)  # Folder name as the label
    return labels

# Face Detection using OpenCV's Haar Cascade
def detect_face(image_path):
    """
    Detect faces in an image using OpenCV's Haar Cascade.
    """
    logging.info(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        logging.warning("No faces detected.")
        return None

    # Assuming only one face for simplicity (the first one detected)
    x, y, w, h = faces[0]
    face = image[y:y + h, x:x + w]
    face_image = Image.fromarray(face).resize(image_size)
    face_array = np.asarray(face_image)
    
    logging.info(f"Face detected and resized.")
    return face_array

# Load the model
def load_model_from_file(model_path):
    """
    Load the trained model from an HDF5 file.
    """
    model = load_model(model_path)
    logging.info("Model loaded from file.")
    return model

# Preprocess image for prediction
def preprocess_image(image):
    """
    Normalize the image for model input.
    """
    image_array = np.array(image).astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Prediction function
def predict(image_path, model, label_encoder):
    """
    Predict the class of the given image using the trained model.
    """
    face = detect_face(image_path)
    if face is None:
        return None

    # Preprocess face image and predict
    face = preprocess_image(face)
    prediction = model.predict(face)
    
    # Get predicted class index
    predicted_class_index = np.argmax(prediction, axis=1)
    # Map class index to name using LabelEncoder
    predicted_class_name = label_encoder.inverse_transform(predicted_class_index)[0]
    
    return predicted_class_name

if __name__ == "__main__":
    # File paths
    model_path = 'face_recognition_model.h5'  # Path to the saved model
    image_path = 'AS1\dataset\Anurag\Anurag _3.jpg'  # Path to the image you want to predict
    dataset_path = 'AS1\dataset'  # Path to the dataset folder (top-level directory)

    # Extract labels from dataset folder
    labels = extract_labels_from_dataset(dataset_path)
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)  # Fit the encoder on the dataset labels

    # Load the trained model
    model = load_model_from_file(model_path)

    # Make a prediction
    predicted_class_name = predict(image_path, model, label_encoder)
    
    if predicted_class_name:
        logging.info(f"Predicted class: {predicted_class_name}")
    else:
        logging.error("No face detected in the image.")
