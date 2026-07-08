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

// Helper to log proctoring event and update session counters/risk level
const logSessionEvent = async (sessionId, studentId, event, confidence) => {
  if (!sessionId || !event) {
    throw new Error("sessionId and event are required");
  }

  const session = await ExamSession.findById(sessionId);
  if (!session) throw new Error("Session not found");

  // Determine severity
  const highSeverityEvents = [
    "MULTIPLE_FACES", "NO_FACE", "PHONE_DETECTED",
    "TAB_SWITCH", "FULLSCREEN_EXIT", "CAMERA_DISCONNECTED",
    "MICROPHONE_DISABLED",
  ];
  const mediumSeverityEvents = [
    "LOOKING_AWAY", "WINDOW_BLUR", "WINDOW_FOCUS_LOST",
    "WINDOW_FOCUS_RESTORED", "EXTENSION_WARNING", "MONITORING_FAILURE",
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

  // Update session counters — WINDOW_FOCUS_RESTORED, MONITORING_FAILURE, etc. are informational
  const nonViolationEvents = [
    "WINDOW_FOCUS_RESTORED", "EXAM_START", "EXAM_END", "CAMERA_GRANTED",
    "MONITORING_FAILURE", "MONITORING_RESTORED"
  ];
  
  const isNonViolation = nonViolationEvents.includes(event) || event.startsWith("MONITORING_FAILURE") || event.startsWith("MONITORING_RESTORED");

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
  const isViolation = !isNonViolation;
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

  return { logged: true, severity, riskLevel };
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

    const result = await logSessionEvent(sessionId, studentId, event, confidence);
    res.json(result);
  } catch (err) {
    console.error("logEvent error:", err);
    res.status(500).json({ error: err.message });
  }
};

// ─── Check AI Service Health ──────────────────────────────────────────────────
exports.checkAiHealth = async (req, res) => {
  try {
    console.log("[AI Health Check] Checking Python AI service health...");
    const aiServiceUrl = process.env.AI_SERVICE_URL || "http://127.0.0.1:8000";
    
    const response = await fetch(`${aiServiceUrl}/health`, { signal: AbortSignal.timeout(3000) });
    if (!response.ok) {
      console.error(`[AI Health Check] AI Service returned non-OK status: ${response.status}`);
      return res.status(502).json({ available: false, modelsLoaded: false, error: `HTTP ${response.status}` });
    }
    
    const data = await response.json();
    const isLoaded = data.models_loaded === true;
    console.log(`[AI Health Check] Python AI service is running. Models loaded: ${isLoaded}`);
    return res.json({ available: true, modelsLoaded: isLoaded });
  } catch (err) {
    console.error("========== AI HEALTH ERROR ==========");
  console.error(err);

  if (err.cause) {
    console.error("CAUSE:");
    console.error(err.cause);
  }

  return res.status(502).json({
    available: false,
    modelsLoaded: false,
    error: err.message,
  });
  }
};

// ─── Analyze Frame ────────────────────────────────────────────────────────────
exports.analyzeFrame = async (req, res) => {
  try {
    const { image, student_id, exam_id, session_id } = req.body;
    const studentId = student_id || req.user.studentId;
    const sessionId = session_id;

    console.log(`[AI Frame Analysis] Backend received frame for student ${studentId}, session ${sessionId || 'none'}`);

    if (!image) {
      console.warn("[AI Frame Analysis] Skipping: No image payload provided");
      return res.status(400).json({ error: "image is required" });
    }

    console.log("[AI Frame Analysis] AI processing frame...");
    const aiServiceUrl = process.env.AI_SERVICE_URL || "http://127.0.0.1:8000";

    const response = await fetch(`${aiServiceUrl}/analyze-frame`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image,
        student_id: studentId,
        exam_id,
        session_id: sessionId,
      }),
    });

    if (!response.ok) {
      console.error(`[AI Frame Analysis] Python AI service returned non-OK status: ${response.status}`);
      return res.status(502).json({ error: `AI service error: HTTP ${response.status}` });
    }

    const data = await response.json();
    console.log(`[AI Frame Analysis] Detection complete: found ${data.face_count} faces, ${data.events?.length || 0} events`);

    // Log detected violations in the database automatically if sessionId is present
    if (sessionId && data.events && data.events.length > 0) {
      for (const ev of data.events) {
        try {
          console.log(`[AI Frame Analysis] Logging event "${ev.event}" to DB...`);
          await logSessionEvent(sessionId, studentId, ev.event, ev.confidence);
        } catch (logErr) {
          console.error(`[AI Frame Analysis] Failed to log event "${ev.event}" to DB:`, logErr.message);
        }
      }
    }

    console.log("[AI Frame Analysis] Result returned to frontend");
    return res.json(data);
  } catch (err) {
    console.error("========== AI FRAME ERROR ==========");
  console.error(err);

  if (err.cause) {
    console.error("CAUSE:");
    console.error(err.cause);
  }

  return res.status(502).json({
    error: `AI service connection failed: ${err.message}`,
  });
  }
};


// ─── End Exam Session ──────────────────────────────────────────────────────────

/**
 * POST /api/session/end
 * Ends the exam: calculates score, generates log file, updates session
 */
exports.endSession = async (req, res) => {
  console.log("\n=== [PROFILING] Exam Submission Started (Backend) ===");
  const totalStart = performance.now();

  try {
    const { sessionId, answers, examId } = req.body;
    const { studentId } = req.user;

    // Critical Task: Save Answers, Save Session, Update DB
    console.time("Save answers");
    console.time("Save session");
    console.time("Update MongoDB");
    const dbStart = performance.now();

    const session = await ExamSession.findById(sessionId);
    if (!session) {
      console.timeEnd("Save answers");
      console.timeEnd("Save session");
      console.timeEnd("Update MongoDB");
      return res.status(404).json({ error: "Session not found" });
    }

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
    console.timeEnd("Save session");

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
    console.timeEnd("Save answers");

    // ── Log exam end ───────────────────────────────────────────────────
    await ProctoringLog.create({
      sessionId,
      studentId,
      examId,
      event: "EXAM_END",
      confidence: 100,
      severity: "Low",
    });
    console.timeEnd("Update MongoDB");

    const dbDuration = performance.now() - dbStart;

    // Send HTTP response immediately
    res.json({
      message: "Exam submitted successfully",
      score,
      totalMarks,
      riskLevel: updatedSession.riskLevel,
    });

    // Execute background tasks asynchronously without blocking the client
    setImmediate(async () => {
      try {
        console.log(`[Background Task] Running non-critical submission tasks for session ${sessionId}...`);
        const bgStart = performance.now();

        // 1. Generate Report
        console.time("Generate report");
        const reportStart = performance.now();
        const logs = await ProctoringLog.find({ sessionId }).sort({ timestamp: 1 });
        await generateLogFile(updatedSession, logs);
        const reportDuration = performance.now() - reportStart;
        console.timeEnd("Generate report");

        // 2. Publish Result (Stub)
        console.time("Publish result");
        const publishStart = performance.now();
        await new Promise((resolve) => setTimeout(resolve, 15)); // mock work
        const publishDuration = performance.now() - publishStart;
        console.timeEnd("Publish result");

        // 3. Send Email (Stub)
        console.time("Send email");
        const emailStart = performance.now();
        await new Promise((resolve) => setTimeout(resolve, 35)); // mock SMTP delay
        const emailDuration = performance.now() - emailStart;
        console.timeEnd("Send email");

        // 4. Cleanup Temporary Files (Stub)
        console.time("Cleanup temporary files");
        const cleanupStart = performance.now();
        await new Promise((resolve) => setTimeout(resolve, 10)); // mock file cleanup
        const cleanupDuration = performance.now() - cleanupStart;
        console.timeEnd("Cleanup temporary files");

        const totalBgDuration = performance.now() - bgStart;
        const totalDuration = performance.now() - totalStart;

        // Print timing summary in backend log
        console.log("\n=== BACKEND TIMING SUMMARY ===");
        console.log(`Save answers: ${dbDuration.toFixed(2)}ms`);
        console.log(`Save session: ${dbDuration.toFixed(2)}ms`);
        console.log(`Update MongoDB: ${dbDuration.toFixed(2)}ms`);
        console.log(`Generate report: ${reportDuration.toFixed(2)}ms`);
        console.log(`Publish result: ${publishDuration.toFixed(2)}ms`);
        console.log(`Send email: ${emailDuration.toFixed(2)}ms`);
        console.log(`Cleanup temporary files: ${cleanupDuration.toFixed(2)}ms`);
        console.log(`Total Background Tasks: ${totalBgDuration.toFixed(2)}ms`);
        console.log(`Total submission processing: ${totalDuration.toFixed(2)}ms`);
        console.log("==============================\n");
      } catch (bgErr) {
        console.error(`[Background Error] Failures in background tasks for session ${sessionId}:`, bgErr);
      }
    });

  } catch (err) {
    console.error("endSession error:", err);
    res.status(500).json({ error: err.message });
  }
};

// ─── Upload Video Recording to Cloudinary ─────────────────────────────────────

/**
 * Helper: Uploads a buffer directly to Cloudinary using upload_chunked_stream.
 * Uses stream piping and proper error handling to prevent uncaught exceptions.
 */
const uploadFromBuffer = (buffer, options) => {
  return new Promise((resolve, reject) => {
    let completed = false;

    const writeStream = cloudinary.uploader.upload_chunked_stream(options, (error, result) => {
      if (completed) return;
      completed = true;
      if (error) {
        reject(error);
      } else {
        resolve(result);
      }
    });

    writeStream.on("error", (err) => {
      if (completed) return;
      completed = true;
      reject(err);
    });

    const { Readable } = require("stream");
    const readStream = new Readable();
    
    readStream.on("error", (err) => {
      if (completed) return;
      completed = true;
      reject(err);
    });

    readStream.push(buffer);
    readStream.push(null);
    readStream.pipe(writeStream);
  });
};

exports.uploadRecording = async (req, res) => {
  console.log(`\n=== [PROFILING] Recording Upload Started: Session ${req.body.sessionId} ===`);
  console.time("Upload recording");
  const start = performance.now();

  try {
    const files = req.files || {};
    const videoFile = files.video?.[0];
    const audioFile = files.audio?.[0];

    if (!videoFile && !audioFile) {
      console.timeEnd("Upload recording");
      return res.status(400).json({ error: "No media file provided" });
    }

    const { sessionId } = req.body;
    const session = await ExamSession.findById(sessionId);
    if (!session) {
      console.timeEnd("Upload recording");
      return res.status(404).json({ error: "Session not found" });
    }

    const updates = {};

    // ── Upload Video ───────────────────────────────────────────────────
    if (videoFile && videoFile.buffer && videoFile.buffer.length > 0) {
      const result = await uploadFromBuffer(videoFile.buffer, {
        resource_type: "video",
        folder: "eyezora_recordings",
        public_id: `${session.studentId}_${session._id}_video`,
        overwrite: true,
        chunk_size: 6000000,
      });
      updates.recordingUrl = result.secure_url;
      updates.recordingPublicId = result.public_id;
    }

    // ── Upload Audio ───────────────────────────────────────────────────
    if (audioFile && audioFile.buffer && audioFile.buffer.length > 0) {
      const result = await uploadFromBuffer(audioFile.buffer, {
        resource_type: "video", // Cloudinary uses "video" resource_type for audio
        folder: "eyezora_recordings",
        public_id: `${session.studentId}_${session._id}_audio`,
        overwrite: true,
        chunk_size: 6000000,
      });
      updates.audioRecordingUrl = result.secure_url;
      updates.audioRecordingPublicId = result.public_id;
    }

    await ExamSession.findByIdAndUpdate(sessionId, updates);

    console.timeEnd("Upload recording");
    const duration = performance.now() - start;
    console.log(`=== BACKEND TIMING SUMMARY: Upload recording: ${(duration / 1000).toFixed(2)}s ===\n`);

    res.json({
      videoUrl: updates.recordingUrl || null,
      audioUrl: updates.audioRecordingUrl || null,
    });
  } catch (err) {
    console.timeEnd("Upload recording");
    console.error("uploadRecording error:", err);
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
