from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from UI.core.security import require_admin
from UI.core.config import templates, ATTENDANCE_DIR
import csv, os
from datetime import datetime

router = APIRouter()

@router.get("/admin")
async def admin_home(request: Request):
    resp = require_admin(request)
    if resp:
        return resp
    return templates.TemplateResponse("face_recognition.html", {"request": request})

@router.get("/admin", response_class=HTMLResponse)
def admin_dashboard(request: Request):
    session = get_session(request)

    if not session["logged_in"] or session["role"] != "admin":
        return RedirectResponse("/login", status_code=302)

    res = (
        supabase
        .table("face_embeddings")
        .select("user_id, person_name")
        .neq("user_id", ADMIN_UUID)
        .execute()
    )

    users = res.data or []

    html = "<h2>Admin Dashboard</h2>"

    if not users:
        html += "<p><b>No users available to message.</b></p>"
        return html

    html += """
    <form action="/chat/admin/send" method="post" enctype="multipart/form-data">
        <label>Select User</label><br>
        <select name="user_id" required>
    """

    seen = set()
    for u in users:
        if u["user_id"] not in seen:
            html += f"<option value='{u['user_id']}'>{u['person_name']}</option>"
            seen.add(u["user_id"])

    html += """
        </select><br><br>
        <textarea name="message" placeholder="Type message"></textarea><br><br>
        <input type="file" name="file"><br><br>
        <button type="submit">Send</button>
    </form>

    <br><a href="/chat/admin">View Chats</a>
    """

    return html

