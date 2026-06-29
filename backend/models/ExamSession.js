const mongoose = require("mongoose");

/**
 * ExamSession — one document per student exam attempt.
 * Tracks all proctoring counters and final recording locations.
 */
const examSessionSchema = new mongoose.Schema(
  {
    studentId: {
      type: String,
      required: true,
      index: true,
    },
    studentName: {
      type: String,
      required: true,
    },
    examId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Exam",
      required: true,
    },
    examTitle: {
      type: String,
      required: true,
    },
    assignmentId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "ExamAssignment",
      default: null,
    },
    startTime: {
      type: Date,
      default: Date.now,
    },
    endTime: {
      type: Date,
    },
    durationSeconds: {
      type: Number,
      default: 0,
    },
    status: {
      type: String,
      enum: ["in_progress", "completed", "flagged", "abandoned"],
      default: "in_progress",
    },

    // ── Violation Counters ─────────────────────────────────────────────
    multipleFacesCount: { type: Number, default: 0 },
    noFaceCount: { type: Number, default: 0 },
    lookingAwayCount: { type: Number, default: 0 },
    phoneDetectedCount: { type: Number, default: 0 },
    suspiciousMovementCount: { type: Number, default: 0 },
    tabSwitchCount: { type: Number, default: 0 },
    windowBlurCount: { type: Number, default: 0 },       // Legacy
    windowFocusLostCount: { type: Number, default: 0 },  // Precise focus loss
    fullscreenExitCount: { type: Number, default: 0 },
    extensionWarningCount: { type: Number, default: 0 },
    totalViolations: { type: Number, default: 0 },

    // ── Risk Assessment ────────────────────────────────────────────────
    riskLevel: {
      type: String,
      enum: ["Low", "Medium", "High"],
      default: "Low",
    },

    // ── Video Recording ────────────────────────────────────────────────
    recordingUrl: {
      type: String,
      default: null,
      // Cloudinary secure URL for video
    },
    recordingPublicId: {
      type: String,
      default: null,
    },

    // ── Audio Recording ────────────────────────────────────────────────
    audioRecordingUrl: {
      type: String,
      default: null,
      // Cloudinary secure URL for audio
    },
    audioRecordingPublicId: {
      type: String,
      default: null,
    },

    // ── Log File ──────────────────────────────────────────────────────
    logFilePath: {
      type: String,
      default: null,
      // e.g. "exam_logs/STU001_EXAM01.txt"
    },

    // ── Score ─────────────────────────────────────────────────────────
    score: { type: Number, default: null },
    totalMarks: { type: Number, default: null },
  },
  { timestamps: true }
);

module.exports = mongoose.model("ExamSession", examSessionSchema);
