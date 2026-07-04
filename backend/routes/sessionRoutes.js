const express = require("express");
const router = express.Router();
const multer = require("multer");
const { verifyToken, requireStudent } = require("../middleware/auth");
const {
  startSession,
  logEvent,
  endSession,
  uploadRecording,
  checkAiHealth,
  analyzeFrame,
} = require("../controllers/sessionController");

// In-memory storage for media uploads (piped directly to Cloudinary)
const storage = multer.memoryStorage();

const upload = multer({
  storage,
  limits: { fileSize: 500 * 1024 * 1024 }, // 500MB max per file
});

// All session routes require student auth
router.use(verifyToken, requireStudent);

router.post("/start", startSession);
router.post("/log", logEvent);
router.post("/end", endSession);
router.get("/ai-health", checkAiHealth);
router.post("/analyze", analyzeFrame);

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
