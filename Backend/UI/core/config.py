import os
from dotenv import load_dotenv
from supabase import create_client
from fastapi.templating import Jinja2Templates

load_dotenv(r"C:\MyDocuments\Attendance system Deepface\.env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase env variables not loaded")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BASE = r"C:\MyDocuments\Attendance system Deepface\Backend"
BASE_F=r"C:\MyDocuments\Attendance system Deepface\Frontend"
UPLOAD_DIR = os.path.join(BASE, "uploads")
ATTENDANCE_DIR = os.path.join(BASE, "attendance_logs")
TEMPLATES_DIR = os.path.join(BASE_F, "templates")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

templates = Jinja2Templates(directory=TEMPLATES_DIR)
