const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");
const { verifyToken, requireStudent } = require("../middleware/auth");
const {
  startSession,
  logEvent,
  endSession,
  uploadRecording,
} = require("../controllers/sessionController");

// Temp storage for media uploads (before Cloudinary)
const storage = multer.diskStorage({
  destination: path.join(__dirname, "../temp_uploads/"),
  filename: (req, file, cb) => {
    const ext = file.fieldname === "audio" ? ".webm" : ".webm";
    cb(null, `${Date.now()}_${file.fieldname}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 500 * 1024 * 1024 }, // 500MB max per file
});

// All session routes require student auth
router.use(verifyToken, requireStudent);

router.post("/start", startSession);
router.post("/log", logEvent);
router.post("/end", endSession);

// Accept both video and audio fields in a single request
router.post(
  "/recording",
  upload.fields([
    { name: "video", maxCount: 1 },
    { name: "audio", maxCount: 1 },
  ]),
  uploadRecording
);

module.exports = router;
