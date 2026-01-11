# Face Recognition Based Attendance System

A secure, GPS-validated, face recognition–based attendance system built using FastAPI, DeepFace (FaceNet), and Supabase (pgvector).  
The system supports admin and user roles, real-time face matching, and an admin dashboard with calendar-style attendance visualization.

---

## 🚀 Features

### 🔐 Authentication & Roles
- Admin and User role separation
- Session-based authentication
- Admin-only access control

### 📷 Face Recognition
- Face detection and embedding using DeepFace (FaceNet)
- Multiple embeddings per user for higher accuracy
- Vector similarity matching using Supabase pgvector
- Threshold-based face matching

### 📍 Location Validation
- GPS-based attendance validation
- Attendance blocked if location permission is denied

### 📝 Attendance Management
- Attendance recorded per day
- Stored in CSV files (Supabase support planned)
- Multiple entries per day supported
- Admin dashboard with calendar-style attendance view

### 👨‍💼 Admin Panel
- Record new persons via webcam
- Rename or delete registered persons
- View user-wise attendance in calendar format
- Admin–user chat system

---

## 🧠 Technology Stack

| Layer | Technology |
|------|-----------|
Backend | FastAPI |
Face Recognition | DeepFace (FaceNet) |
Database | Supabase (PostgreSQL + pgvector) |
Attendance Storage | CSV (current) |
Frontend | HTML, CSS, JavaScript |
Face Detection | OpenCV |
Authentication | FastAPI Sessions |

---

## 📁 Project Structure

