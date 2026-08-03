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
- Voice recording
- Tab switch alert

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
- MongoDB Atlas
- JWT Authentication
- Nodemailer
- Multer
- Cloudinary

### AI Service
- FastAPI
- Python
- OpenCV
- YOLOv8 / YOLOv11
- PyTorch

---

# 📂 Project Structure

```
EyeZora/
│
├── frontend/          # Next.js Frontend (App Router + TypeScript)
│   ├── app/
│   ├── lib/
│   ├── public/
│   ├── Dockerfile
│   └── package.json
│
├── backend/           # Node.js + Express API
│   ├── routes/
│   ├── controllers/
│   ├── models/
│   ├── middleware/
│   ├── Dockerfile
│   └── package.json
│
├── ai-services/       # FastAPI + YOLO AI Proctoring Service
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── *.pt           # YOLO model weights
│
├── docker-compose.yml # Orchestrates all three services
└── README.md
```

---

# 🐳 Docker Quick Start (Recommended)

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Git

## 1. Clone the Repository

```bash
git clone https://github.com/Rifana1977/EyeZora.git
cd EyeZora
```

## 2. Configure Environment Variables

The backend `.env` file is already present at `backend/.env`. Verify it contains your real credentials:

```env
MONGO_URI=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/
JWT_SECRET=your_jwt_secret_here
PORT=5000
FRONTEND_URL=http://localhost:3000

CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

EMAIL_FROM=your@gmail.com
EMAIL_PASS=your_app_password

RESET_TOKEN_EXPIRY_HOURS=2
AI_SERVICE_URL=http://ai-services:8000
```

> **Note**: `AI_SERVICE_URL=http://ai-services:8000` is the Docker service name. Do not change this value when running with Docker Compose.

## 3. Build and Start All Services

```bash
docker compose up --build
```

This command will:
1. Build the `ai-services` image (installs PyTorch, YOLO, OpenCV)
2. Build the `backend` image (installs Node.js packages)
3. Build the `frontend` image (compiles Next.js)
4. Start all three services on the shared `eyezora-net` Docker network

> ⏳ **First build takes 5–15 minutes** due to PyTorch and YOLO downloads (~2GB). Subsequent builds are fast due to Docker layer caching.

## 4. Access the Application

| Service    | URL                    |
|------------|------------------------|
| Frontend   | http://localhost:3000  |
| Backend    | http://localhost:5000  |
| AI Service | http://localhost:8000  |

## 5. Seed the Admin Account (first time only)

```bash
docker compose exec backend node scripts/seedAdmin.js
```

Default credentials:
- **Email**: `admin@eyezora.com`
- **Password**: `Admin@123`

> ⚠️ Change this password immediately after first login.

---

# 🐳 Docker Commands Reference

### Start all services (foreground)
```bash
docker compose up --build
```

### Start all services (background / detached)
```bash
docker compose up --build -d
```

### View live logs (all services)
```bash
docker compose logs -f
```

### View logs for a specific service
```bash
docker compose logs -f backend
docker compose logs -f ai-services
docker compose logs -f frontend
```

### Stop all services
```bash
docker compose down
```

### Restart a single service
```bash
docker compose restart backend
docker compose restart ai-services
docker compose restart frontend
```

### Rebuild a single service after code changes
```bash
docker compose up --build backend
```

### Open a shell inside a container
```bash
docker compose exec backend sh
docker compose exec ai-services bash
```

---

# ⚙️ Manual Installation (without Docker)

## Prerequisites

- Node.js 20+
- Python 3.11+
- Git

## 1. Clone Repository

```bash
git clone https://github.com/Rifana1977/EyeZora.git
cd EyeZora
```

## 2. Start AI Service

```bash
cd ai-services
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## 3. Start Backend

```bash
cd backend
npm install
npm run dev
```

## 4. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

---

# 🔑 Environment Variables

Create / update `backend/.env`:

```env
MONGO_URI=
JWT_SECRET=
EMAIL_FROM=
EMAIL_PASS=

CLOUDINARY_CLOUD_NAME=
CLOUDINARY_API_KEY=
CLOUDINARY_API_SECRET=

AI_SERVICE_URL=http://127.0.0.1:8000    # local dev
# AI_SERVICE_URL=http://ai-services:8000  # Docker
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
- Non-root Docker containers

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
