const mongoose = require("mongoose");

/**
 * ProctoringLog — one document per detected violation event.
 * Stored in real-time as the student takes the exam.
 */
const proctoringLogSchema = new mongoose.Schema({
  sessionId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "ExamSession",
    required: true,
    index: true,
  },
  studentId: {
    type: String,
    required: true,
    index: true,
  },
  examId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Exam",
    required: true,
  },
  timestamp: {
    type: Date,
    default: Date.now,
  },
  event: {
    type: String,
    required: true,
    enum: [
      // ── AI Detection Events ──────────────────────
      "MULTIPLE_FACES",
      "NO_FACE",
      "LOOKING_AWAY",
      "PHONE_DETECTED",
      "SUSPICIOUS_MOVEMENT",
      // ── Browser Monitoring Events ────────────────
      "TAB_SWITCH",
      "WINDOW_BLUR",            // Legacy — kept for backward compatibility
      "WINDOW_FOCUS_LOST",      // Precise: window lost focus (alt-tab, OS switch)
      "WINDOW_FOCUS_RESTORED",  // Precise: window regained focus
      "FULLSCREEN_EXIT",
      "EXTENSION_WARNING",
      // ── Camera & Audio Events ────────────────────
      "CAMERA_DISCONNECTED",
      "CAMERA_GRANTED",
      "MICROPHONE_DISABLED",
      "MONITORING_FAILURE",
      "MONITORING_RESTORED",
      // ── Session Lifecycle ────────────────────────
      "EXAM_START",
      "EXAM_END",
    ],
  },
  confidence: {
    type: Number, // 0-100
    default: 100,
  },
  severity: {
    type: String,
    enum: ["Low", "Medium", "High"],
    default: "High",
  },
});

module.exports = mongoose.model("ProctoringLog", proctoringLogSchema);
