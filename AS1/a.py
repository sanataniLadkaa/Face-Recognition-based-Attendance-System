from deepface import DeepFace
from pymongo import MongoClient
from gridfs import GridFS
import os
import io
from PIL import Image
import pandas as pd
import numpy as np

# MongoDB Setup
client = MongoClient("mongodb://localhost:27017/")
db = client["image_database"]
fs = GridFS(db)

# Paths
test_image_path = r"C:\MyDocuments\ASystem\AS1\AbhishekPIc1-removebg-preview.jpg"
representations_path = "representations_vggface.pkl"  # Path to store embeddings

# Threshold for matching
threshold = 0.4  # Lower threshold = stricter match


def preprocess_image(image):
    """Preprocess images before generating embeddings."""
    try:
        # Resize image to match the input size of the model
        image = image.resize((224, 224))
        return image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def get_images_from_gridfs():
    """Retrieve and preprocess all images from MongoDB GridFS."""
    images = []
    for grid_out in fs.find():
        try:
            # Read image data
            image = Image.open(io.BytesIO(grid_out.read()))
            image.filename = grid_out.filename  # Save the original filename
            processed_image = preprocess_image(image)
            if processed_image:
                images.append(processed_image)
        except Exception as e:
            print(f"Error processing {grid_out.filename}: {e}")
    return images


def update_face_database():
    """Build or update the face database."""
    print("Updating face database...")
    images = get_images_from_gridfs()
    embeddings = []

    for image in images:
        try:
            image.save("temp.jpg")  # Temporarily save image
            embedding = DeepFace.represent(img_path="temp.jpg", model_name="VGG-Face", enforce_detection=False)
            for item in embedding:
                embeddings.append({
                    "identity": image.filename,
                    "embedding": item["embedding"]
                })
        except Exception as e:
            print(f"Error generating embedding for {image.filename}: {e}")

    # Save to local file
    if embeddings:
        df = pd.DataFrame(embeddings)
        df.to_pickle(representations_path)
        print("Face database updated successfully!")
    else:
        print("No valid images found.")


def recognize_face():
    """Perform face recognition on a test image."""
    if not os.path.exists(test_image_path):
        print(f"Error: The file '{test_image_path}' does not exist.")
        return

    print("Recognizing the face...")
    test_embedding = DeepFace.represent(
        img_path=test_image_path, 
        model_name="VGG-Face", 
        enforce_detection=False
    )[0]['embedding']

    if not os.path.exists(representations_path):
        print(f"Error: Database embeddings not found at '{representations_path}'.")
        return

    # Load embeddings from database
    db_embeddings = pd.read_pickle(representations_path)

    # Compare test image embedding with database
    best_match = None
    best_distance = float("inf")

    for index, row in db_embeddings.iterrows():
        db_embedding = row['embedding']
        distance = np.linalg.norm(np.array(test_embedding) - np.array(db_embedding))  # Euclidean distance
        if distance < best_distance and distance <= threshold:
            best_distance = distance
            best_match = row['identity']

    if best_match:
        print(f"Recognized: {best_match} with distance {best_distance}")
    else:
        print("No match found within the threshold.")


# Update database and recognize face
update_face_database()
recognize_face()
