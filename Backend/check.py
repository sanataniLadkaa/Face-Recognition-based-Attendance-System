from deepface import DeepFace
import os

# Paths
dataset_path = r"C:\MyDocuments\Attendance system Deepface\Backend\dataset"
input_image_path = r"C:\MyDocuments\Attendance system Deepface\Backend\WhatsApp Image 2024-06-25 at 16.43.14_8db353d0-Photoroom.jpg"

# Model and detector
model_name = 'Facenet'
detector_backend = 'opencv'

# Prepare: Load image lists for all folders first
folder_images = {}  # {person_folder: [list of image paths]}
max_images = 0

for person_folder in os.listdir(dataset_path):
    person_folder_path = os.path.join(dataset_path, person_folder)
    if not os.path.isdir(person_folder_path):
        continue

    images = os.listdir(person_folder_path)
    image_paths = [os.path.join(person_folder_path, img) for img in images if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    folder_images[person_folder] = image_paths
    max_images = max(max_images, len(image_paths))

# Cross-wise comparison
found_person = None

for i in range(max_images):  # i = index of image per folder
    for person_folder, images in folder_images.items():
        if i >= len(images):
            continue  # This folder has fewer images

        image_path = images[i]
        print(f"🔍 Comparing with: {person_folder} → {os.path.basename(image_path)}")

        try:
            result = DeepFace.verify(
                img1_path=input_image_path,
                img2_path=image_path,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False,
                distance_metric='cosine'
            )

            if result["verified"]:
                print(f"✅ Match found with: {person_folder}")
                print(f"Matching Image: {image_path}")
                found_person = person_folder
                break

        except Exception as e:
            print(f"⚠️ Error comparing with {image_path}: {e}")

    if found_person:
        break  # Stop after first overall match

if found_person:
    print(f"\n🎉 Final recognized person: {found_person}")
else:
    print("\n❌ No match found for the given image.")
