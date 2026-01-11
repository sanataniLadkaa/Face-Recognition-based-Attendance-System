# login.py

# ADMIN user_id
ADMIN_USER_ID = "f2bf951d-f86b-4041-ac0c-19332450f8ee"

# passwords (temporary / MVP)
ADMIN_PASSWORD = "admin123"
USER_PASSWORD = "user123"

# all allowed users
ALLOWED_USERS = {
    "4b34ef01-d395-40a9-af48-53fd644fc8ba",
    "6ce64e5a-9ff8-4669-9d11-bf141ba6237f",
    "7ff14cd4-4ca0-430e-a63f-4797a782a813",
    "e3943426-6ece-4a54-9954-3dd20f95797d",
    "f2bf951d-f86b-4041-ac0c-19332450f8ee",
}


def verify_login(user_id: str, password: str):
    # check user exists
    if user_id not in ALLOWED_USERS:
        return None

    # admin check
    if user_id == ADMIN_USER_ID:
        if password == ADMIN_PASSWORD:
            return "admin"
        return None

    # normal user
    if password == USER_PASSWORD:
        return "user"

    return None
