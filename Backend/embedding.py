from deepface import DeepFace
import os
import pickle

dataset_path = r"C:\MyDocuments\Attendance system Deepface\Backend\dataset"
embedding_file = r"C:\MyDocuments\Attendance system Deepface\Backend\embeddings.pkl"

model_name = 'Facenet'
detector_backend = 'opencv'

embeddings = []

for person_folder in os.listdir(dataset_path):
    person_folder_path = os.path.join(dataset_path, person_folder)

    if not os.path.isdir(person_folder_path):
        continue

    for image_name in os.listdir(person_folder_path):
        image_path = os.path.join(person_folder_path, image_name)

        try:
            representation = DeepFace.represent(
                img_path=image_path,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False
            )

            embedding_vector = representation[0]["embedding"]

            embeddings.append({
                "person": person_folder,
                "image_path": image_path,
                "embedding": embedding_vector
            })

            print(f"✅ Embedded: {image_path}")

        except Exception as e:
            print(f"⚠️ Error with {image_path}: {e}")

# Save all embeddings
with open(embedding_file, "wb") as f:
    pickle.dump(embeddings, f)

print("\n🎉 All dataset embeddings saved successfully!")
