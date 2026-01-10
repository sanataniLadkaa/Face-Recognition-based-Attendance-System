from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
import os, csv

from Backend.UI.core.security import require_admin, get_session
from Backend.UI.core.config import templates, ATTENDANCE_DIR

router = APIRouter()


# ================= ADMIN HOME =================
@router.get("/admin", response_class=HTMLResponse)
async def admin_home(request: Request):
    resp = require_admin(request)
    if resp:
        return resp

    return templates.TemplateResponse(
        "face_recognition.html",
        {"request": request}
    )


# ================= ADMIN DASHBOARD =================
@router.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    session = get_session(request)

    if not session or session["role"] != "admin":
        return RedirectResponse("/login", status_code=302)

    attendance = []

    if os.path.exists(ATTENDANCE_DIR):
        for file in os.listdir(ATTENDANCE_DIR):
            if file.endswith("_attendance.csv"):
                date = file.replace("_attendance.csv", "")
                with open(os.path.join(ATTENDANCE_DIR, file)) as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        attendance.append({
                            "date": date,
                            "name": row[0],
                            "time": row[1]
                        })

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "attendance": attendance
        }
    )
