from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from Backend.UI.store import router as attendance_router
from Backend.UI.routes import auth, admin, user, recognition, file_upload

app = FastAPI()

# Redirect root to /login
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/login")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(user.router)
app.include_router(recognition.router)
app.include_router(attendance_router, prefix="/attendance")
app.include_router(file_upload.router)
# app.include_router(admin.router)
# app.include_router(recognition.router)
