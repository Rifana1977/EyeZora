const ExamSession = require("../models/ExamSession");
const ProctoringLog = require("../models/ProctoringLog");
const Question = require("../models/Question");
const Submission = require("../models/Submission");
const ExamAssignment = require("../models/ExamAssignment");
const cloudinary = require("cloudinary").v2;
const fs = require("fs");
const path = require("path");

// Configure Cloudinary
cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
});

// ─── Start Exam Session ────────────────────────────────────────────────────────

/**
 * POST /api/session/start
 * Creates a new exam session for the logged-in student
 */
exports.startSession = async (req, res) => {
  try {
    const { examId, examTitle, assignmentId } = req.body;
    const { studentId, name } = req.user;

    // Check for an existing in-progress session (prevent duplicate)
    const existing = await ExamSession.findOne({
      studentId,
      examId,
      status: "in_progress",
    });

    if (existing) {
      return res.json({ sessionId: existing._id, resumed: true });
    }

    const session = await ExamSession.create({
      studentId,
      studentName: name,
      examId,
      examTitle,
      assignmentId: assignmentId || null,
    });

    // Mark assignment as started
    if (assignmentId) {
      await ExamAssignment.findByIdAndUpdate(assignmentId, { status: "started" });
    }

    // Log exam start event
    await ProctoringLog.create({
      sessionId: session._id,
      studentId,
      examId,
      event: "EXAM_START",
      confidence: 100,
      severity: "Low",
    });

    res.status(201).json({ sessionId: session._id, resumed: false });
  } catch (err) {
    console.error("startSession error:", err);
    res.status(500).json({ error: err.message });
  }
};

// ─── Log Proctoring Event ──────────────────────────────────────────────────────

/**
 * POST /api/session/log
 * Logs a proctoring violation event and updates session counters
 */
exports.logEvent = async (req, res) => {
  try {
    const { sessionId, event, confidence } = req.body;
    const { studentId } = req.user;

    if (!sessionId || !event) {
      return res.status(400).json({ error: "sessionId and event are required" });
    }

    const session = await ExamSession.findById(sessionId);
    if (!session) return res.status(404).json({ error: "Session not found" });

    // Determine severity
    const highSeverityEvents = [
      "MULTIPLE_FACES", "NO_FACE", "PHONE_DETECTED",
      "TAB_SWITCH", "FULLSCREEN_EXIT", "CAMERA_DISCONNECTED",
      "MICROPHONE_DISABLED",
    ];
    const mediumSeverityEvents = [
      "LOOKING_AWAY", "WINDOW_BLUR", "WINDOW_FOCUS_LOST",
      "WINDOW_FOCUS_RESTORED", "EXTENSION_WARNING",
    ];
    const severity = highSeverityEvents.includes(event)
      ? "High"
      : mediumSeverityEvents.includes(event)
      ? "Medium"
      : "Low";

    // Create log entry
    await ProctoringLog.create({
      sessionId,
      studentId,
      examId: session.examId,
      event,
      confidence: confidence || 100,
      severity,
    });

    // Update session counters — WINDOW_FOCUS_RESTORED is informational (no violation count)
    const nonViolationEvents = ["WINDOW_FOCUS_RESTORED", "EXAM_START", "EXAM_END", "CAMERA_GRANTED"];
    const counterMap = {
      MULTIPLE_FACES: "multipleFacesCount",
      NO_FACE: "noFaceCount",
      LOOKING_AWAY: "lookingAwayCount",
      PHONE_DETECTED: "phoneDetectedCount",
      SUSPICIOUS_MOVEMENT: "suspiciousMovementCount",
      TAB_SWITCH: "tabSwitchCount",
      WINDOW_BLUR: "windowBlurCount",
      WINDOW_FOCUS_LOST: "windowFocusLostCount",
      FULLSCREEN_EXIT: "fullscreenExitCount",
      EXTENSION_WARNING: "extensionWarningCount",
    };

    const counterField = counterMap[event];
    const isViolation = !nonViolationEvents.includes(event);
    const updateQuery = isViolation ? { $inc: { totalViolations: 1 } } : { $inc: {} };

    if (counterField) {
      updateQuery.$inc[counterField] = 1;
    }

    const updatedSession = await ExamSession.findByIdAndUpdate(
      sessionId,
      updateQuery,
      { new: true }
    );

    // Recompute risk level
    const violations = updatedSession.totalViolations;
    const riskLevel =
      violations <= 2 ? "Low" : violations <= 5 ? "Medium" : "High";

    await ExamSession.findByIdAndUpdate(sessionId, { riskLevel });

    res.json({ logged: true, severity, riskLevel });
  } catch (err) {
    console.error("logEvent error:", err);
    res.status(500).json({ error: err.message });
  }
};

// ─── End Exam Session ──────────────────────────────────────────────────────────

/**
 * POST /api/session/end
 * Ends the exam: calculates score, generates log file, updates session
 */
exports.endSession = async (req, res) => {
  try {
    const { sessionId, answers, examId } = req.body;
    const { studentId } = req.user;

    const session = await ExamSession.findById(sessionId);
    if (!session) return res.status(404).json({ error: "Session not found" });

    const endTime = new Date();
    const durationSeconds = Math.floor(
      (endTime - session.startTime) / 1000
    );

    // ── Score Calculation ──────────────────────────────────────────────
    const questions = await Question.find({ examId });
    let score = 0;
    let totalMarks = 0;

    questions.forEach((q, i) => {
      totalMarks += q.marks || 1;
      if (answers && answers[i] === q.correctOptionIndex) {
        score += q.marks || 1;
      }
    });

    // ── Update Session ─────────────────────────────────────────────────
    const updatedSession = await ExamSession.findByIdAndUpdate(
      sessionId,
      {
        status: session.totalViolations >= 6 ? "flagged" : "completed",
        endTime,
        durationSeconds,
        score,
        totalMarks,
      },
      { new: true }
    );

    // ── Mark Assignment as Completed ───────────────────────────────────
    if (session.assignmentId) {
      await ExamAssignment.findByIdAndUpdate(session.assignmentId, {
        status: "completed",
      });
    }

    // ── Save Submission ────────────────────────────────────────────────
    await Submission.create({
      examId,
      studentId,
      examSessionId: sessionId,
      answers: answers || [],
      score,
      totalMarks,
    });

    // ── Generate Text Log File ────────────────────────────────────────
    const logs = await ProctoringLog.find({ sessionId }).sort({ timestamp: 1 });
    await generateLogFile(updatedSession, logs);

    // ── Log exam end ───────────────────────────────────────────────────
    await ProctoringLog.create({
      sessionId,
      studentId,
      examId,
      event: "EXAM_END",
      confidence: 100,
      severity: "Low",
    });

    res.json({
      message: "Exam submitted successfully",
      score,
      totalMarks,
      riskLevel: updatedSession.riskLevel,
    });
  } catch (err) {
    console.error("endSession error:", err);
    res.status(500).json({ error: err.message });
  }
};

// ─── Upload Video Recording to Cloudinary ─────────────────────────────────────

/**
 * POST /api/session/recording
 * Receives a video blob, uploads to Cloudinary, stores URL in session
 */
exports.uploadRecording = async (req, res) => {
  try {
    const files = req.files || {};
    const videoFile = files.video?.[0];
    const audioFile = files.audio?.[0];

    if (!videoFile && !audioFile) {
      return res.status(400).json({ error: "No media file provided" });
    }

    const { sessionId } = req.body;
    const session = await ExamSession.findById(sessionId);
    if (!session) return res.status(404).json({ error: "Session not found" });

    const updates = {};

    // ── Upload Video ───────────────────────────────────────────────────
    if (videoFile) {
      try {
        const result = await cloudinary.uploader.upload(videoFile.path, {
          resource_type: "video",
          folder: "eyezora_recordings",
          public_id: `${session.studentId}_${session._id}_video`,
          overwrite: true,
        });
        updates.recordingUrl = result.secure_url;
        updates.recordingPublicId = result.public_id;
      } finally {
        if (fs.existsSync(videoFile.path)) fs.unlinkSync(videoFile.path);
      }
    }

    // ── Upload Audio ───────────────────────────────────────────────────
    if (audioFile) {
      try {
        const result = await cloudinary.uploader.upload(audioFile.path, {
          resource_type: "video", // Cloudinary uses "video" resource_type for audio
          folder: "eyezora_recordings",
          public_id: `${session.studentId}_${session._id}_audio`,
          overwrite: true,
        });
        updates.audioRecordingUrl = result.secure_url;
        updates.audioRecordingPublicId = result.public_id;
      } finally {
        if (fs.existsSync(audioFile.path)) fs.unlinkSync(audioFile.path);
      }
    }

    await ExamSession.findByIdAndUpdate(sessionId, updates);

    res.json({
      videoUrl: updates.recordingUrl || null,
      audioUrl: updates.audioRecordingUrl || null,
    });
  } catch (err) {
    console.error("uploadRecording error:", err);
    // Clean up any leftover temp files
    const files = req.files || {};
    for (const fileList of Object.values(files)) {
      for (const f of fileList) {
        if (f?.path && fs.existsSync(f.path)) fs.unlinkSync(f.path);
      }
    }
    res.status(500).json({ error: err.message });
  }
};

// ─── Helper: Generate Text Log File ───────────────────────────────────────────

async function generateLogFile(session, logs) {
  try {
    const dir = path.join(__dirname, "../exam_logs");
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

    const fileName = `${session.studentId}_${session._id}.txt`;
    const filePath = path.join(dir, fileName);

    const eventLabel = (event) => {
      const labels = {
        MULTIPLE_FACES: "Multiple Faces Detected",
        NO_FACE: "No Face Detected",
        LOOKING_AWAY: "Looking Away",
        PHONE_DETECTED: "Phone Detected",
        SUSPICIOUS_MOVEMENT: "Suspicious Movement",
        TAB_SWITCH: "Browser Tab Switch",
        WINDOW_BLUR: "Window Focus Lost (Legacy)",
        WINDOW_FOCUS_LOST: "Window Focus Lost",
        WINDOW_FOCUS_RESTORED: "Window Focus Restored",
        FULLSCREEN_EXIT: "Fullscreen Exit",
        EXTENSION_WARNING: "Extension Warning",
        CAMERA_DISCONNECTED: "Camera Disconnected",
        CAMERA_GRANTED: "Camera Access Granted",
        EXAM_START: "Exam Started",
        EXAM_END: "Exam Ended",
      };
      return labels[event] || event;
    };

    const lines = [
      "═══════════════════════════════════════════════════════════════",
      "                    EYEZORA EXAM PROCTORING LOG                ",
      "═══════════════════════════════════════════════════════════════",
      `Student ID   : ${session.studentId}`,
      `Student Name : ${session.studentName}`,
      `Exam         : ${session.examTitle}`,
      `Session ID   : ${session._id}`,
      `Start Time   : ${session.startTime?.toISOString()}`,
      `End Time     : ${session.endTime?.toISOString()}`,
      `Duration     : ${Math.floor(session.durationSeconds / 60)} min ${session.durationSeconds % 60} sec`,
      `Risk Level   : ${session.riskLevel}`,
      "",
      "─── VIOLATION SUMMARY ──────────────────────────────────────────",
      `Multiple Faces      : ${session.multipleFacesCount}`,
      `No Face Detected    : ${session.noFaceCount}`,
      `Looking Away        : ${session.lookingAwayCount}`,
      `Phone Detected      : ${session.phoneDetectedCount}`,
      `Suspicious Movement : ${session.suspiciousMovementCount}`,
      `Tab Switches        : ${session.tabSwitchCount}`,
      `Window Focus Lost   : ${session.windowFocusLostCount}`,
      `Full-Screen Exits   : ${session.fullscreenExitCount}`,
      `Extension Warnings  : ${session.extensionWarningCount}`,
      `TOTAL VIOLATIONS    : ${session.totalViolations}`,
      "",
      "─── EVENT LOG ──────────────────────────────────────────────────",
      "Timestamp                  | Event                     | Conf | Severity",
      "───────────────────────────|───────────────────────────|──────|─────────",
    ];

    logs.forEach((log) => {
      const ts = new Date(log.timestamp).toLocaleString("en-IN", {
        timeZone: "Asia/Kolkata",
      });
      const event = eventLabel(log.event).padEnd(25);
      const conf = `${log.confidence}%`.padEnd(4);
      lines.push(`${ts.padEnd(27)}| ${event} | ${conf} | ${log.severity}`);
    });

    lines.push("═══════════════════════════════════════════════════════════════");

    fs.writeFileSync(filePath, lines.join("\n"), "utf-8");

    // Update logFilePath in session
    await ExamSession.findByIdAndUpdate(session._id, {
      logFilePath: `exam_logs/${fileName}`,
    });
  } catch (err) {
    console.error("generateLogFile error:", err);
  }
}
