Face Recognition Based Attendance System

A secure, GPS-validated, face recognition–based attendance system built using FastAPI, DeepFace (FaceNet), and Supabase (pgvector).
The system supports admin and user roles, real-time face matching, controlled attendance marking, and an admin dashboard with calendar-style attendance visualization.

🚀 Features
🔐 Authentication & Roles

Admin and User role separation

Session-based authentication

Admin-only access control for management features

📷 Face Recognition

Face detection and embedding using DeepFace (FaceNet)

Multiple embeddings per user for better accuracy

Fast similarity matching using Supabase pgvector

Threshold-based matching to prevent false positives

📍 Location Validation

Attendance allowed only within permitted GPS coordinates

Browser-based geolocation verification

Attendance blocked if GPS permission is denied

📝 Attendance Management

Attendance recorded per day

Stored in CSV files (Supabase support planned)

Multiple entries per day supported

Admin dashboard with:

User list

Month/day grid calendar

Present / Absent / Future day states

Hover-based time details

👨‍💼 Admin Panel

Record new persons via webcam

Rename or delete registered persons

View attendance per user in calendar format

Live admin–user chat support

🧠 Technology Stack
Layer	Technology
Backend	FastAPI
Face Recognition	DeepFace (FaceNet)
Database	Supabase (PostgreSQL + pgvector)
Attendance Storage	CSV (current), Supabase (planned)
Frontend	HTML, CSS, JavaScript
Face Detection	OpenCV
Auth & Sessions	FastAPI sessions
Deployment Ready	Yes (path-safe, env-based config)
📁 Project Structure
Backend/
├── UI/
│   ├── routes/
│   │   ├── admin.py
│   │   ├── recognition.py
│   │   ├── attendance.py
│   ├── core/
│   │   ├── config.py
│   │   ├── security.py
│   │   ├── location.py
│   ├── templates/
│   │   ├── face_recognition.html
│   │   ├── dashboard.html
│   │   ├── result.html
│   ├── static/
│
├── temp_capture/
├── attendance/
├── uploads/
├── main.py

⚙️ Setup Instructions
1️⃣ Clone Repository
git clone https://github.com/<your-username>/Face-Recognition-based-Attendance-System.git
cd Face-Recognition-based-Attendance-System

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Configure Environment Variables

Create a .env file:

SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

🧪 Supabase Requirements
Table: face_embeddings
Column	Type
user_id	UUID
person_name	TEXT
embedding	VECTOR(512)
RPC Function: match_face_embedding

Used for fast vector similarity matching using pgvector.

▶️ Run the Application
uvicorn main:app --reload


Open in browser:

http://127.0.0.1:8000

📊 Attendance Logic

Attendance is recorded once per face match

Stored per day in:

attendance/YYYY-MM-DD_attendance.csv


Admin dashboard:

Green → Present

Light Green → Single entry

Red → Absent

Grey → Future dates

🔒 Security Considerations

Admin-only routes protected via session validation

Users cannot mark attendance for others

GPS spoofing prevention via browser permission enforcement

Stable identity ensured via user_id

🛠️ Future Enhancements

Migrate attendance storage from CSV → Supabase

Average embeddings per user

Liveness detection

Face anti-spoofing

Attendance export (PDF / Excel)

Analytics dashboard

👤 Author

Anurag Tiwari
AI / Machine Learning Developer

📄 License

This project is licensed for educational and research purposes.
