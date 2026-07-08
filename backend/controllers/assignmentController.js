const ExamAssignment = require("../models/ExamAssignment");
const Student = require("../models/Student");
const Exam = require("../models/Exam");
const ExamSession = require("../models/ExamSession");
const Submission = require("../models/Submission");
const ProctoringLog = require("../models/ProctoringLog");
const cloudinary = require("cloudinary").v2;
const fs = require("fs");
const path = require("path");

cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
});



// ─── Helper: compute computed status from DB status + times ───────────────────

function computeStatus(assignment) {
  if (["completed", "cancelled", "started"].includes(assignment.status)) {
    return assignment.status;
  }
  const now = new Date();
  if (now > assignment.endTime) return "expired";
  if (now >= assignment.startTime) return "upcoming"; // within window
  return assignment.status; // "assigned"
}

// ─── Get All Assignments ───────────────────────────────────────────────────────

/**
 * GET /api/admin/assignments
 * Query params: search, examId, status, page, limit
 */
exports.getAssignments = async (req, res) => {
  try {
    const {
      search = "",
      examId,
      status,
      page = 1,
      limit = 20,
    } = req.query;

    const skip = (parseInt(page) - 1) * parseInt(limit);

    // Build filter
    const filter = {};
    if (examId) filter.examId = examId;
    if (status && status !== "all") filter.status = status;

    // If searching, get matching student IDs first
    let studentIdFilter = null;
    if (search) {
      const matchingStudents = await Student.find({
        $or: [
          { name: { $regex: search, $options: "i" } },
          { studentId: { $regex: search, $options: "i" } },
          { email: { $regex: search, $options: "i" } },
        ],
      }).select("studentId");
      studentIdFilter = matchingStudents.map((s) => s.studentId);
      if (studentIdFilter.length > 0) {
        filter.studentId = { $in: studentIdFilter };
      } else if (search) {
        // No matching students — return empty
        return res.json({ assignments: [], total: 0, page: parseInt(page), totalPages: 0 });
      }
    }

    const total = await ExamAssignment.countDocuments(filter);
    const assignments = await ExamAssignment.find(filter)
      .populate("studentObjectId", "studentId name email isActive")
      .populate("examId", "title duration")
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(parseInt(limit));

    // Enrich with computed status
    const enriched = assignments.map((a) => {
      const obj = a.toObject();
      obj.computedStatus = computeStatus(a);
      return obj;
    });

    res.json({
      assignments: enriched,
      total,
      page: parseInt(page),
      totalPages: Math.ceil(total / parseInt(limit)),
    });
  } catch (err) {
    console.error("getAssignments error:", err);
    res.status(500).json({ error: err.message });
  }
};

// ─── Create Assignment ─────────────────────────────────────────────────────────

/**
 * POST /api/admin/assignments
 * Body: { studentObjectId, examId, startTime, endTime, duration, notes }
 */
exports.createAssignment = async (req, res) => {
  try {
    const { studentObjectId, examId, startTime, duration, notes } = req.body;
    let { endTime } = req.body;

    if (!studentObjectId || !examId || !startTime || !duration) {
      return res.status(400).json({ error: "studentObjectId, examId, startTime, and duration are required" });
    }

    // Validate student
    const student = await Student.findById(studentObjectId);
    if (!student) return res.status(404).json({ error: "Student not found" });

    // Validate exam
    const exam = await Exam.findById(examId);
    if (!exam) return res.status(404).json({ error: "Exam not found" });

    if (!endTime) {
      endTime = new Date(new Date(startTime).getTime() + Number(duration) * 60 * 1000);
    }

    // Conflict detection: check if this student has overlapping assignment for ANY exam
    const conflict = await ExamAssignment.findOne({
      studentId: student.studentId,
      status: { $in: ["assigned", "upcoming", "started"] },
      $or: [
        { startTime: { $lt: new Date(endTime) }, endTime: { $gt: new Date(startTime) } },
      ],
    });
    if (conflict) {
      return res.status(409).json({
        error: `Scheduling conflict: student already has an assignment from ${new Date(conflict.startTime).toLocaleString("en-IN", { timeZone: "Asia/Kolkata" })} to ${new Date(conflict.endTime).toLocaleString("en-IN", { timeZone: "Asia/Kolkata" })}`,
      });
    }

    const assignment = await ExamAssignment.create({
      studentId: student.studentId,
      studentObjectId,
      examId,
      startTime: new Date(startTime),
      endTime: new Date(endTime),
      duration: Number(duration),
      notes: notes || "",
      assignedBy: req.user?.email || "admin",
    });

    const populated = await ExamAssignment.findById(assignment._id)
      .populate("studentObjectId", "studentId name email isActive")
      .populate("examId", "title duration");

    const obj = populated.toObject();
    obj.computedStatus = computeStatus(populated);

    res.status(201).json(obj);
  } catch (err) {
    console.error("createAssignment error:", err);
    res.status(500).json({ error: err.message });
  }
};

// ─── Update Assignment ─────────────────────────────────────────────────────────

/**
 * PUT /api/admin/assignments/:id
 * Body: { examId, startTime, endTime, duration, notes }
 */
exports.updateAssignment = async (req, res) => {
  try {
    const { id } = req.params;
    const { examId, startTime, duration, notes } = req.body;
    let { endTime } = req.body;

    const assignment = await ExamAssignment.findById(id);
    if (!assignment) return res.status(404).json({ error: "Assignment not found" });

    if (assignment.status === "completed" || assignment.status === "cancelled") {
      return res.status(400).json({ error: "Cannot modify a completed or cancelled assignment" });
    }

    if (examId) {
      const exam = await Exam.findById(examId);
      if (!exam) return res.status(404).json({ error: "Exam not found" });
    }

    const updates = {};
    if (examId) updates.examId = examId;
    if (startTime) updates.startTime = new Date(startTime);
    if (duration) updates.duration = Number(duration);
    if (notes !== undefined) updates.notes = notes;

    // Recalculate endTime if startTime or duration changed and no explicit endTime was passed
    if (startTime || duration) {
      const finalStartTime = startTime ? new Date(startTime) : new Date(assignment.startTime);
      const finalDuration = duration ? Number(duration) : assignment.duration;
      updates.endTime = new Date(finalStartTime.getTime() + finalDuration * 60 * 1000);
    } else if (endTime) {
      updates.endTime = new Date(endTime);
    }

    const updated = await ExamAssignment.findByIdAndUpdate(id, updates, { new: true })
      .populate("studentObjectId", "studentId name email isActive")
      .populate("examId", "title duration");

    const obj = updated.toObject();
    obj.computedStatus = computeStatus(updated);

    res.json(obj);
  } catch (err) {
    console.error("updateAssignment error:", err);
    res.status(500).json({ error: err.message });
  }
};

// ─── Cancel Assignment ─────────────────────────────────────────────────────────

/**
 * DELETE /api/admin/assignments/:id
 * Soft-delete: sets status to "cancelled"
 */
exports.cancelAssignment = async (req, res) => {
  try {
    const { id } = req.params;
    const assignment = await ExamAssignment.findById(id);
    if (!assignment) return res.status(404).json({ error: "Assignment not found" });

    if (assignment.status === "completed") {
      return res.status(400).json({ error: "Cannot cancel a completed assignment" });
    }

    await ExamAssignment.findByIdAndUpdate(id, { status: "cancelled" });
    res.json({ message: "Assignment cancelled successfully" });
  } catch (err) {
    console.error("cancelAssignment error:", err);
    res.status(500).json({ error: err.message });
  }
};

// ─── Get Active Assignment for Student ────────────────────────────────────────

/**
 * GET /api/admin/assignments/student/:studentId
 * Returns the current/latest active assignment for a student
 */
exports.getStudentAssignment = async (req, res) => {
  try {
    const { studentId } = req.params;

    const assignment = await ExamAssignment.findOne({
      studentId,
      status: { $in: ["assigned", "upcoming", "started"] },
    })
      .populate("examId", "title duration isActive")
      .sort({ createdAt: -1 });

    if (!assignment) {
      return res.json({ assignment: null });
    }

    const obj = assignment.toObject();
    obj.computedStatus = computeStatus(assignment);

    res.json({ assignment: obj });
  } catch (err) {
    console.error("getStudentAssignment error:", err);
    res.status(500).json({ error: err.message });
  }
};

// ─── Bulk Create Assignments ───────────────────────────────────────────────────

/**
 * POST /api/admin/assignments/bulk
 * Body: { studentObjectIds: [], examId, startTime, duration, notes }
 * Assigns the same exam+window to multiple students.
 * Returns per-student success/failure.
 */
exports.bulkCreateAssignment = async (req, res) => {
  try {
    const { studentObjectIds, examId, startTime, duration, notes } = req.body;

    if (!studentObjectIds || !Array.isArray(studentObjectIds) || studentObjectIds.length === 0) {
      return res.status(400).json({ error: "studentObjectIds array is required" });
    }
    if (!examId || !startTime || !duration) {
      return res.status(400).json({ error: "examId, startTime, and duration are required" });
    }

    const exam = await Exam.findById(examId);
    if (!exam) return res.status(404).json({ error: "Exam not found" });

    const endTime = new Date(new Date(startTime).getTime() + Number(duration) * 60 * 1000);

    const results = [];
    const errors = [];

    for (const studentObjectId of studentObjectIds) {
      try {
        const student = await Student.findById(studentObjectId);
        if (!student) {
          errors.push({ studentObjectId, error: "Student not found" });
          continue;
        }

        // Conflict detection: check if this student has overlapping assignment for ANY exam
        const conflict = await ExamAssignment.findOne({
          studentId: student.studentId,
          status: { $in: ["assigned", "upcoming", "started"] },
          $or: [
            { startTime: { $lt: endTime }, endTime: { $gt: new Date(startTime) } },
          ],
        });

        if (conflict) {
          errors.push({
            studentObjectId,
            studentId: student.studentId,
            studentName: student.name,
            error: `Scheduling conflict: already has an assignment from ${conflict.startTime.toISOString()} to ${conflict.endTime.toISOString()}`,
          });
          continue;
        }

        const assignment = await ExamAssignment.create({
          studentId: student.studentId,
          studentObjectId,
          examId,
          startTime: new Date(startTime),
          endTime,
          duration: Number(duration),
          notes: notes || "",
          assignedBy: req.user?.email || "admin",
        });

        results.push({
          studentObjectId,
          studentId: student.studentId,
          studentName: student.name,
          assignmentId: assignment._id,
          status: "assigned",
        });
      } catch (err) {
        errors.push({ studentObjectId, error: err.message });
      }
    }

    res.status(201).json({
      assigned: results.length,
      failed: errors.length,
      results,
      errors,
    });
  } catch (err) {
    console.error("bulkCreateAssignment error:", err);
    res.status(500).json({ error: err.message });
  }
};

// ─── Bulk Cancel Assignments ───────────────────────────────────────────────────

/**
 * DELETE /api/admin/assignments/bulk-cancel
 * Body: { ids: string[] }
 * Soft-delete (status → "cancelled") for multiple assignments.
 * Skips any assignments that are already "completed".
 * Returns { cancelled, skipped, requested }.
 */
exports.bulkCancelAssignments = async (req, res) => {
  try {
    const { ids } = req.body;
    if (!ids || !Array.isArray(ids) || ids.length === 0) {
      return res.status(400).json({ error: "ids array is required" });
    }

    // Only cancel non-completed assignments
    const result = await ExamAssignment.updateMany(
      { _id: { $in: ids }, status: { $ne: "completed" } },
      { status: "cancelled" }
    );

    const cancelled = result.modifiedCount;
    const skipped = ids.length - cancelled;

    res.json({ cancelled, skipped, requested: ids.length });
  } catch (err) {
    console.error("bulkCancelAssignments error:", err);
    res.status(500).json({ error: err.message });
  }
};

// Helper to perform cascade delete of all data associated with an assignment
const performCascadeDelete = async (assignmentId) => {
  try {
    const assignment = await ExamAssignment.findById(assignmentId);
    const student = assignment ? `${assignment.studentId}` : "Unknown";
    const exam = assignment ? `${assignment.examId}` : "Unknown";

    const sessions = await ExamSession.find({ assignmentId });
    const sessionIds = sessions.map(s => s._id.toString());

    console.log(`\n=== [CASCADE DELETE] Starting deletion for assignment ${assignmentId} ===`);
    console.log(`Student: ${student}`);
    console.log(`Exam: ${exam}`);

    let cloudinaryCount = 0;
    let localFileCount = 0;

    for (const session of sessions) {
      const sessionId = session._id;

      // 1. Delete video and audio recordings from Cloudinary
      if (session.recordingPublicId) {
        try {
          await cloudinary.uploader.destroy(session.recordingPublicId, { resource_type: "video" });
          console.log(`[Cloudinary] Deleted video recording: ${session.recordingPublicId}`);
          cloudinaryCount++;
        } catch (err) {
          console.error(`[Cloudinary] Failed to delete video recording ${session.recordingPublicId}:`, err.message);
        }
      }
      if (session.audioRecordingPublicId) {
        try {
          await cloudinary.uploader.destroy(session.audioRecordingPublicId, { resource_type: "video" });
          console.log(`[Cloudinary] Deleted audio recording: ${session.audioRecordingPublicId}`);
          cloudinaryCount++;
        } catch (err) {
          console.error(`[Cloudinary] Failed to delete audio recording ${session.audioRecordingPublicId}:`, err.message);
        }
      }

      // 2. Delete text log file from local filesystem
      if (session.logFilePath) {
        try {
          const fullPath = path.join(__dirname, "../", session.logFilePath);
          if (fs.existsSync(fullPath)) {
            fs.unlinkSync(fullPath);
            console.log(`[Local Disk] Deleted proctoring log file: ${fullPath}`);
            localFileCount++;
          }
        } catch (err) {
          console.error(`[Local Disk] Failed to delete log file for session ${sessionId}:`, err.message);
        }
      }

      // 3. Delete associated ProctoringLog events
      const deletedLogs = await ProctoringLog.deleteMany({ sessionId });
      console.log(`[Database] Deleted ${deletedLogs.deletedCount} proctoring logs for session ${sessionId}`);

      // 4. Delete associated Submission (report)
      const deletedSubmissions = await Submission.deleteMany({ examSessionId: sessionId });
      console.log(`[Database] Deleted ${deletedSubmissions.deletedCount} submissions (reports) for session ${sessionId}`);

      // 5. Delete the ExamSession itself
      await ExamSession.findByIdAndDelete(sessionId);
      console.log(`[Database] Deleted ExamSession document: ${sessionId}`);
    }

    // 6. Delete the ExamAssignment document itself
    const deletedAssignment = await ExamAssignment.findByIdAndDelete(assignmentId);
    console.log(`[Database] Deleted ExamAssignment document: ${assignmentId}`);
    
    console.log(`=== [CASCADE DELETE] Successfully completed for assignment ${assignmentId} ===\n`);
    return {
      sessionsCount: sessions.length,
      cloudinaryCount,
      localFileCount
    };
  } catch (err) {
    console.error(`[Cascade Delete Error] Failed for assignment ${assignmentId}:`, err);
    throw err;
  }
};

// ─── Delete Single Assignment (Hard Delete with Cascade) ──────────────────────
/**
 * DELETE /api/admin/assignments/:id
 * Hard deletes assignment and all associated sessions, reports, logs, and cloud recordings.
 */
exports.deleteAssignment = async (req, res) => {
  try {
    const { id } = req.params;

    // Perform cascade delete
    await performCascadeDelete(id);

    res.json({ message: "Assignment and all related proctoring sessions, reports, logs, and media recordings deleted successfully." });
  } catch (err) {
    console.error("deleteAssignment error:", err);
    res.status(500).json({ error: err.message });
  }
};

// ─── Delete Multiple Assignments (Bulk Delete with Cascade) ───────────────────
/**
 * DELETE /api/admin/assignments
 * Body: { assignmentIds: string[] }
 * Hard deletes multiple assignments and all associated sessions, reports, logs, and recordings.
 */
exports.bulkDeleteAssignments = async (req, res) => {
  try {
    const { assignmentIds } = req.body;
    const ids = assignmentIds || req.body.ids;

    if (!ids || !Array.isArray(ids) || ids.length === 0) {
      return res.status(400).json({ error: "assignmentIds array is required." });
    }

    // Perform cascade delete for each assignment
    for (const id of ids) {
      await performCascadeDelete(id);
    }

    res.json({
      message: `${ids.length} assignment(s) and all related sessions, reports, logs, and recordings deleted successfully.`,
      deleted: ids.length,
      requested: ids.length,
    });
  } catch (err) {
    console.error("bulkDeleteAssignments error:", err);
    res.status(500).json({ error: err.message });
  }
};

