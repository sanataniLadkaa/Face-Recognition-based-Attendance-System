from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from UI.store import router as attendance_router
from UI.routes import auth, admin, user, recognition, file_upload

app = FastAPI()

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
