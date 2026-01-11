import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
from fastapi.templating import Jinja2Templates

# Load env (local only; Render injects env vars automatically)
load_dotenv()

# Supabase config
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase env variables not loaded")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Project base directory (Docker-safe)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# This resolves to: /app/Backend

UPLOAD_DIR = BASE_DIR / "uploads"
ATTENDANCE_DIR = BASE_DIR / "attendance_logs"
TEMPLATES_DIR = BASE_DIR.parent / "Frontend" / "templates"

# Ensure folders exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)

# Jinja templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
