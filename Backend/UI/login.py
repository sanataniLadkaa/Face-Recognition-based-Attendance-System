import json
import os

CREDENTIAL_FILE = r"C:\MyDocuments\Attendance system Deepface\Backend\secrets.json"

def verify_login(username, password):
    with open(CREDENTIAL_FILE, "r") as f:
        data = json.load(f)

    for user in data["users"]:
        if user["username"] == username and user["password"] == password:
            return True

    return False
