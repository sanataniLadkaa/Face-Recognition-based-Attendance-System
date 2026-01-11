import pickle
from supabase import create_client
from dotenv import load_dotenv
import os

# Load env
load_dotenv(r"C:\MyDocuments\Attendance system Deepface\.env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load embeddings.pkl
pkl_path = r"C:\MyDocuments\Attendance system Deepface\Backend\embeddings.pkl"

with open(pkl_path, "rb") as f:
    embeddings = pickle.load(f)

print(f"📂 Total embeddings: {len(embeddings)}")

# Insert into Supabase
for i, record in enumerate(embeddings):
    data = {
        "person_name": record["person"],
        "embedding": record["embedding"],
        "image_path": record.get("image_path", "")
    }

    supabase.table("face_embeddings").insert(data).execute()

    if i % 50 == 0:
        print(f"✅ Uploaded {i} embeddings")

print("🎉 All embeddings uploaded to Supabase")
