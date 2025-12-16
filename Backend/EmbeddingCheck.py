from deepface import DeepFace
import numpy as np
import pickle

# Paths
embedding_file = r"C:\MyDocuments\Attendance system Deepface\Backend\embeddings.pkl"
input_image_path = r"C:\MyDocuments\Attendance system Deepface\Backend\WhatsApp Image 2024-06-25 at 16.43.14_8db353d0-Photoroom.jpg"

model_name = 'Facenet'
detector_backend = 'opencv'
distance_threshold = 0.6  # Cosine threshold

# Load saved embeddings
with open(embedding_file, "rb") as f:
    saved_embeddings = pickle.load(f)

# Organize embeddings folder-wise (person-wise)
folder_embeddings = {}  # { person_name: [embedding1, embedding2, ...] }

for record in saved_embeddings:
    person = record["person"]
    embedding = record["embedding"]

    if person not in folder_embeddings:
        folder_embeddings[person] = []
    
    folder_embeddings[person].append(embedding)

# Build model
model = DeepFace.build_model(model_name)

# Get embedding of input image
try:
    input_representation = DeepFace.represent(
        img_path=input_image_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=False
    )
    input_embedding = input_representation[0]["embedding"]

except Exception as e:
    print(f"⚠️ Error extracting embedding for input image: {e}")
    exit()

# Find max number of images per folder (max embedding length)
max_embeddings = max(len(emb_list) for emb_list in folder_embeddings.values())

found_person = None
min_distance = float('inf')

# Cross-wise comparison: check i-th embedding from each folder
for i in range(max_embeddings):
    for person, embeddings in folder_embeddings.items():
        if i >= len(embeddings):
            continue  # This person has fewer images

        db_embedding = embeddings[i]

        # Cosine distance
        cosine_distance = np.dot(input_embedding, db_embedding) / (np.linalg.norm(input_embedding) * np.linalg.norm(db_embedding))
        cosine_distance = 1 - cosine_distance

        print(f"🔍 Comparing with: {person} (Image {i+1}), Distance: {cosine_distance:.4f}")

        if cosine_distance < distance_threshold:
            found_person = person
            min_distance = cosine_distance
            print(f"✅ Match found: {found_person} (Distance: {min_distance:.4f})")
            break  # Stop inner loop (found match in this pass)

    if found_person:
        break  # Stop outer loop (no need to check further passes)

if found_person:
    print(f"\n🎉 Final recognized person: {found_person}")
else:
    print("\n❌ No match found for the given image.")
