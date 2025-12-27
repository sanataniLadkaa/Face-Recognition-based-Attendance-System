from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, JSONResponse
from core.security import get_session
from core.config import templates, ATTENDANCE_DIR
import csv, os

router = APIRouter()

@router.get("/user")
async def user_dashboard(request: Request):
    s = get_session(request)
    if not s["logged_in"] or s["role"] != "user":
        return RedirectResponse("/login")

    return templates.TemplateResponse("user_dashboard.html", {"request": request, "username": s["username"]})

@router.get("/attendance/my")
async def my_attendance(request: Request):
    s = get_session(request)
    if not s["logged_in"] or s["role"] != "user":
        return JSONResponse(status_code=403, content={"error": "Unauthorized"})

    records = []
    for file in os.listdir(ATTENDANCE_DIR):
        with open(os.path.join(ATTENDANCE_DIR, file)) as f:
            reader = csv.reader(f)
            next(reader, None)
            for r in reader:
                if r[0] == s["username"]:
                    records.append({"date": file[:10], "time": r[1]})
    return records
