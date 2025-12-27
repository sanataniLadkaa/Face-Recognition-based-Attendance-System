from fastapi import Request
from fastapi.responses import RedirectResponse

def get_session(request: Request):
    return {
        "logged_in": request.cookies.get("logged_in") == "yes",
        "role": request.cookies.get("role"),
        "username": request.cookies.get("username"),
        "user_id": request.cookies.get("user_id"),
    }

def require_login(request: Request):
    if not get_session(request)["logged_in"]:
        return RedirectResponse("/login")

def require_admin(request: Request):
    s = get_session(request)
    if not s["logged_in"] or s["role"] != "admin":
        return RedirectResponse("/login")
