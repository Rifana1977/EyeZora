<img width="1630" height="192" alt="image" src="https://github.com/user-attachments/assets/168fd154-e491-4583-a3bb-8b4958b80f4e" />

# 👁️ EyeZora

**EyeZora** is an AI-powered online examination and intelligent remote proctoring system that enables educational institutions to conduct secure online examinations with real-time monitoring, automated violation detection, and comprehensive reporting.

---

## 🚀 Features

### 👨‍🎓 Student Portal
- Secure authentication using Student ID or Email
- Forgot password with email-based password reset
- Fullscreen enforced examination
- Live webcam monitoring
- Automatic exam submission on timeout
- Real-time AI violation alerts
- voice recording
- tab switch alert

### 👨‍💼 Admin Portal
- Student management
- Exam creation and scheduling
- Assign exams to students
- Live monitoring dashboard
- Examination reports
- Publish results
- Bulk management (students, assignments, reports)

### 🤖 AI Proctoring
- Face detection
- No-face detection
- Multiple face detection
- Mobile phone detection
- Real-time violation logging

### ⚡ Performance
- Asynchronous report generation
- Background email delivery
- Cloudinary video uploads
- Fast exam submission (<450 ms)
- Automatic cleanup of related records during deletion

---

# 🛠 Tech Stack

### Frontend
- Next.js (App Router)
- React
- TypeScript
- CSS Modules

### Backend
- Node.js
- Express.js
- MongoDB
- JWT Authentication
- Nodemailer
- Multer

### AI Service
- FastAPI
- Python
- OpenCV
- YOLOv11
- MediaPipe
- PyTorch

---

# 📂 Project Structure

```
EyeZora/
│
├── frontend/          # Next.js Frontend
├── backend/           # Express Backend APIs
│   ├── routes/
│   ├── controllers/
│   ├── models/
│   └── main.py        # FastAPI AI Service
└── README.md
```

---

# ⚙️ Installation

## Prerequisites

- Node.js 18+
- Python 3.10+
- MongoDB
- Git

---

## 1. Clone Repository

```bash
git clone https://github.com/Rifana1977/EyeZora.git
cd EyeZora
```

---

## 2. Start AI Service

```bash
cd backend

pip install -r requirements.txt

uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 3. Start Backend

```bash
cd backend

npm install

npm run dev
```

---

## 4. Start Frontend

```bash
cd frontend

npm install

npm run dev
```

---

# 🔑 Environment Variables

Create a `.env` file inside the `backend` directory.

```env
MONGO_URI=
JWT_SECRET=
EMAIL_USER=
EMAIL_PASS=

CLOUDINARY_CLOUD_NAME=
CLOUDINARY_API_KEY=
CLOUDINARY_API_SECRET=

AI_SERVICE_URL=http://127.0.0.1:8000
```

---

# 🧪 Testing

Run the integration test:

```bash
cd backend

node test-flow.js
```

---

# 📸 Screenshots

> Add screenshots or GIFs of:
- Student Login
- Admin Dashboard
- AI Monitoring
- Reports
- Live Exam Page

---

# 🔒 Security Features

- JWT Authentication
- Role-Based Access Control
- Password Reset via Email
- Fullscreen Enforcement
- AI-Based Proctoring
- Session Monitoring
- Secure Password Hashing

---

# 📌 Future Improvements

- Voice activity detection
- Liveness detection
- AI cheating risk score
- Analytics dashboard
- Multi-camera support

# 📄 License

This project is developed for educational purposes.

<img width="1610" height="26" alt="image" src="https://github.com/user-attachments/assets/25f06dcc-26d9-49ce-bf97-95db5994a37e" />

