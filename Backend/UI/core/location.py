import math

# ================= OFFICE CONFIG =================
OFFICE_LATITUDE = 27.1751       # 🔁 set office latitude
OFFICE_LONGITUDE = 78.0421     # 🔁 set office longitude
MAX_DISTANCE_METERS = 10000000        # allowed radius

# ================= DISTANCE ======================
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius (meters)

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ================= VALIDATION ====================
def validate_user_location(latitude, longitude):
    """
    Returns:
        (True, None)  -> inside office
        (False, reason) -> outside / error
    """

    if latitude is None or longitude is None:
        return False, "Location permission required (GPS not received)"

    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except ValueError:
        return False, "Invalid GPS coordinates"

    distance = haversine_distance(
        latitude,
        longitude,
        OFFICE_LATITUDE,
        OFFICE_LONGITUDE
    )

    if distance > MAX_DISTANCE_METERS:
        return False, "You are not inside office premises"

    return True, None
