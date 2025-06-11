import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

from os import listdir
from os.path import isdir, join
from PIL import Image
from numpy import savez_compressed, asarray
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
import logging
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import logging
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
image_size = (224, 224)
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
confidence_threshold = 0.6

# Face Detection using MTCNN
def detect_faces(image_path):
    """
    Detect faces in an image using MTCNN.
    """
    logging.info(f"Processing image: {image_path}")
    try:
        image = Image.open(image_path).convert('RGB')
        pixels = asarray(image)
    except Exception as e:
        logging.error(f"Error loading image: {e}")
        return []

    detector = MTCNN()
    results = detector.detect_faces(pixels)

    if not results:
        logging.warning("No faces detected.")
        return []

    faces = []
    for i, result in enumerate(results):
        x1, y1, width, height = result['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        face_image = Image.fromarray(face).resize(image_size)
        faces.append(asarray(face_image))
        logging.info(f"Face {i+1} detected and resized.")

    return faces


# Dataset Preparation
def load_faces(directory):
    """
    Load and preprocess all faces from a directory.
    """
    faces = []
    for filename in listdir(directory):
        path = join(directory, filename)
        detected_faces = detect_faces(path)
        faces.extend(detected_faces)
    return faces


def load_dataset(directory):
    """
    Load dataset with faces and labels from subdirectories.
    """
    X, y = [], []
    for subdir in listdir(directory):
        path = join(directory, subdir)
        if not isdir(path):
            logging.warning(f"Skipping non-directory: {path}")
            continue
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        X.extend(faces)
        y.extend(labels)
        logging.info(f"Loaded {len(faces)} faces for class '{subdir}'.")
    return np.asarray(X), np.asarray(y)


# Preprocess Faces for Model Input
def preprocess_faces(faces):
    """
    Normalize faces for model input.
    """
    faces_array = np.array(faces).astype('float32') / 255.0
    return faces_array

num_classes = 5
# Model Definition
def create_vgg19_model(num_classes, learning_rate=0.0001):
    """
    Create and compile a VGG19-based model.
    """
    base_model = VGG19(weights="imagenet", include_top=False, input_shape=(*image_size, 3))
    for layer in base_model.layers:
        layer.trainable = False  # Freeze the base model

    model = Sequential([
    base_model,  # Use a pretrained model and freeze some layers
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(),
    Dropout(0.3),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Dataset Preparation (with Label Encoding)
def load_single_dataset(directory):
    """
    Load dataset from a single directory where subdirectories represent labels.
    """
    X, y = [], []
    for subdir in listdir(directory):
        path = directory + '/' + subdir + '/'
        if not isdir(path):
            logging.warning(f"Skipping non-directory: {path}")
            continue
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        X.extend(faces)
        y.extend(labels)
        logging.info(f"Loaded {len(faces)} faces for class '{subdir}'.")
    return np.asarray(X), np.asarray(y)


if __name__ == "__main__":

    # Single dataset directory
    dataset_directory = 'AS1\dataset'  # Replace with your dataset folder path

    # Load dataset
    X, y = load_single_dataset(dataset_directory)

    # Encode labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # Transform string labels to integers

    # Save the dataset for reuse
    savez_compressed('single_dataset_encoded.npz', X, y_encoded)

    # Normalize dataset
    X = preprocess_faces(X)

    # Split into training and validation sets
    trainX, testX, trainy, testy = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Oversampling using SMOTE for class imbalance
    smote = SMOTE(sampling_strategy={
    0: len(trainy[trainy == 2]),  
    1: len(trainy[trainy == 2]),   
    3: len(trainy[trainy == 2]),
    4: len(trainy[trainy == 2])
})  # Balances all classes to the size of the largest class
 # Oversample class 1 (Shivangi Singh)
    trainX_flattened = trainX.reshape(trainX.shape[0], -1)  # SMOTE requires 2D data
    trainX_resampled, trainy_resampled = smote.fit_resample(trainX_flattened, trainy)

    # Reshape back to original dimensions after SMOTE
    trainX_resampled = trainX_resampled.reshape(-1, trainX.shape[1], trainX.shape[2], trainX.shape[3])

    logging.info(f"Original Training Dataset Size: {trainX.shape[0]}")
    logging.info(f"Resampled Training Dataset Size: {trainX_resampled.shape[0]}")

    # Prepare the model
    num_classes = len(label_encoder.classes_)  # Number of unique labels
    model = create_vgg19_model(num_classes)

    # Train the model
    logging.info("Training the model...")
    model.fit(trainX_resampled, trainy_resampled, epochs=20, validation_data=(testX, testy))

    # Save the trained model
    model.save("face_recognition_model.h5")
    logging.info("Model saved: face_recognition_modelN.h5")
