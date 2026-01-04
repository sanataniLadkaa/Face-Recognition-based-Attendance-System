from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse
from UI.login import verify_login
from UI.core.config import templates, supabase

router = APIRouter()

def get_username_from_user_id(user_id: str):
    res = (
        supabase
        .table("face_embeddings")
        .select("person_name")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    return res.data[0]["person_name"] if res.data else "User"

@router.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/login")
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    role = verify_login(username, password)
    if not role:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})

    response = RedirectResponse("/admin" if role == "admin" else "/user", status_code=302)

    response.set_cookie("logged_in", "yes", httponly=True)
    response.set_cookie("role", role, httponly=True)
    response.set_cookie("user_id", username, httponly=True)
    response.set_cookie("username", get_username_from_user_id(username), httponly=True)

    return response

@router.get("/logout")
async def logout():
    r = RedirectResponse("/login")
    for k in ["logged_in", "role", "username", "user_id"]:
        r.delete_cookie(k)
    return r
