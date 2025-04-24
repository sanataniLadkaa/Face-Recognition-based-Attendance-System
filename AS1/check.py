from pymongo import MongoClient
from gridfs import GridFS
import os
from PIL import Image
import io

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["image_database"]
fs = GridFS(db)

# Output dataset directory
dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)

# Fetch images from MongoDB and save to dataset
for grid_out in fs.find():
    label = grid_out.metadata.get("label", "unknown").strip()  # Strip extra spaces from label
    person_dir = os.path.join(dataset_dir, label)
    os.makedirs(person_dir, exist_ok=True)

    # Get file path and ensure it doesn't have any unintended spaces
    file_name = grid_out.filename.strip()  # Strip extra spaces from filename
    file_path = os.path.join(person_dir, file_name)
    
    # Save image
    with open(file_path, "wb") as f:
        f.write(grid_out.read())
    
    print(f"Image saved at: {file_path}")  # Print the file path for confirmation

print(f"Images saved to {dataset_dir} directory.")
