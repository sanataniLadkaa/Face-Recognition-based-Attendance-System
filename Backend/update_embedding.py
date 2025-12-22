import os
import pickle
from deepface import DeepFace

# Paths
dataset_dir = r"C:\MyDocuments\Attendance system Deepface\Backend\dataset"
embedding_file = r"C:\MyDocuments\Attendance system Deepface\Backend\embeddings.pkl"

model_name = "Facenet"
detector_backend = "opencv"

# Load existing embeddings
if os.path.exists(embedding_file):
    with open(embedding_file, "rb") as f:
        saved_embeddings = pickle.load(f)
else:
    saved_embeddings = []

# Track already processed images
processed_images = {record["image_path"] for record in saved_embeddings}

print("Already processed images:", processed_images)

# Load model once
model = DeepFace.build_model(model_name)

new_embeddings = []

# Scan dataset folder:
for person_folder in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person_folder)

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        # Skip already processed images
        if img_path in processed_images:
            continue

        print(f"Processing new image: {img_path}")

        try:
            representation = DeepFace.represent(
                img_path=img_path,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False
            )

            embedding = representation[0]["embedding"]

            new_embeddings.append({
                "person": person_folder,
                "embedding": embedding,
                "image_path": img_path
            })

        except Exception as e:
            print(f"⚠️ Could not process {img_path}: {e}")

# Save updated embeddings
all_embeddings = saved_embeddings + new_embeddings

with open(embedding_file, "wb") as f:
    pickle.dump(all_embeddings, f)

print(f"\n🎉 Added {len(new_embeddings)} new embeddings.")
print("Total embeddings:", len(all_embeddings))
