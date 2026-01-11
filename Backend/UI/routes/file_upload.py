from fastapi import APIRouter, UploadFile, File, Request, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from supabase import create_client
from dotenv import load_dotenv
import os, uuid, shutil
from datetime import datetime

load_dotenv()
router = APIRouter()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ADMIN_UUID = "f2bf951d-f86b-4041-ac0c-19332450f8ee"

ALLOWED_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".txt",
    ".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mp3", ".wav", ".zip"
}

def get_session(request: Request):
    return {
        "logged_in": request.cookies.get("logged_in") == "yes",
        "role": request.cookies.get("role"),
        "user_id": request.cookies.get("user_id")
    }

def save_file(file: UploadFile):
    if not file or not file.filename:
        return None, None, None   # ← allow no file

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return None, None, "Invalid file type"

    filename = f"{uuid.uuid4()}{ext}"
    path = os.path.join(UPLOAD_DIR, filename)

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return path, file.content_type, None


# ================= ADMIN USERS =================

@router.get("/admin/users")
def fetch_users(request: Request):
    session = get_session(request)

    if not session["logged_in"] or session["role"] != "admin":
        return []

    res = (
        supabase
        .table("face_embeddings")
        .select("user_id, person_name")
        .neq("user_id", ADMIN_UUID)
        .execute()
    )

    # 🔑 DEDUPLICATE BY user_id
    users = {}
    for row in res.data or []:
        uid = row["user_id"]
        if uid not in users:
            users[uid] = row["person_name"]

    return [
        {"id": uid, "name": name}
        for uid, name in users.items()
    ]




# ================= ADMIN SEND =================

@router.post("/chat/admin/send")
async def admin_send_chat(
    request: Request,
    user_id: str = Form(...),
    message: str = Form(None),
    file: UploadFile = File(None)
):
    session = get_session(request)

    if not session.get("logged_in") or session.get("role") != "admin":
        return RedirectResponse("/login", status_code=302)

    if not user_id:
        return HTMLResponse("Invalid user", status_code=400)

    file_url, file_type = None, None
    if file and file.filename:
        file_url, file_type, error = save_file(file)
        if error:
            return HTMLResponse(error, status_code=400)

    supabase.table("chats").insert({
        "sender_id": session["user_id"],
        "receiver_id": user_id,
        "sender_role": "admin",
        "message": message,
        "file_url": file_url,
        "file_type": file_type,
        "created_at": datetime.utcnow().isoformat()
    }).execute()

    return {"status": "sent"}

# ================= USER SEND =================

@router.post("/chat/user/send")
async def user_send_chat(
    request: Request,
    message: str = Form(None),
    file: UploadFile = File(None)
):
    session = get_session(request)
    if session["role"] != "user":
        return RedirectResponse("/login", 302)

    file_url, file_type = None, None
    if file:
        file_url, file_type, err = save_file(file)
        if err:
            return HTMLResponse(err, 400)

    supabase.table("chats").insert({
        "sender_id": session["user_id"],
        "receiver_id": ADMIN_UUID,
        "sender_role": "user",
        "message": message,
        "file_url": file_url,
        "file_type": file_type,
        "created_at": datetime.utcnow().isoformat()
    }).execute()

    return RedirectResponse("/user", 302)


# ================= USER HISTORY (FIXED) =================

@router.get("/chat/user/history")
def user_chat_history(request: Request):
    session = get_session(request)

    if not session["logged_in"] or session["role"] != "user":
        return []

    uid = session["user_id"]

    res = (
        supabase
        .table("chats")
        .select("*")
        .or_(f"sender_id.eq.{uid},receiver_id.eq.{uid}")
        .order("created_at")
        .execute()
    )

    chats = []
    for c in res.data or []:
        chats.append({
            "id": c["id"],
            "message": c["message"],
            "file_url": c["file_url"],
            "created_at": c["created_at"],
            "sender_role": c["sender_role"],
            # ✅ FIX: explicit name
            "sender_name": "You" if c["sender_role"] == "user" else "Admin"
        })

    return chats

@router.get("/chat/admin/history/{user_id}")
def admin_chat_history(request: Request, user_id: str):
    session = get_session(request)

    if not session["logged_in"] or session["role"] != "admin":
        return []

    admin_id = session["user_id"]

    res = (
        supabase
        .table("chats")
        .select("*")
        .or_(
            f"and(sender_id.eq.{admin_id},receiver_id.eq.{user_id}),"
            f"and(sender_id.eq.{user_id},receiver_id.eq.{admin_id})"
        )
        .order("created_at")
        .execute()
    )

    chats = []
    for c in res.data or []:
        chats.append({
            "id": c["id"],
            "message": c["message"],
            "file_url": c["file_url"],
            "file_type": c["file_type"],
            "created_at": c["created_at"],
            "sender_role": c["sender_role"]
        })

    return chats

