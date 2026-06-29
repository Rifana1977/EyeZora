const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const Admin = require("../models/Admin");
const Student = require("../models/Student");
const ExamAssignment = require("../models/ExamAssignment");
const { JWT_SECRET } = require("../middleware/auth");

// ─── Admin Login ───────────────────────────────────────────────────────────────

/**
 * POST /api/auth/admin/login
 * Body: { email, password }
 */
exports.adminLogin = async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ error: "Email and password are required" });
    }

    const admin = await Admin.findOne({ email: email.toLowerCase() });
    if (!admin) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    const isMatch = await bcrypt.compare(password, admin.passwordHash);
    if (!isMatch) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    const token = jwt.sign(
      { id: admin._id, name: admin.name, email: admin.email, role: "admin" },
      JWT_SECRET,
      { expiresIn: "8h" }
    );

    return res.json({
      token,
      user: { id: admin._id, name: admin.name, email: admin.email, role: "admin" },
    });
  } catch (err) {
    console.error("adminLogin error:", err);
    return res.status(500).json({ error: "Server error" });
  }
};

// ─── Student Login ─────────────────────────────────────────────────────────────

/**
 * POST /api/auth/student/login
 * Body: { identifier, password }  — identifier = studentId OR email
 *
 * Returns the student user object + their current active assignment (if any).
 * Login is allowed even if no assignment exists.
 */
exports.studentLogin = async (req, res) => {
  try {
    const { identifier, password } = req.body;

    if (!identifier || !password) {
      return res
        .status(400)
        .json({ error: "Identifier and password are required" });
    }

    // Allow login by studentId or email
    const student = await Student.findOne({
      $or: [
        { studentId: identifier },
        { email: identifier.toLowerCase() },
      ],
    });

    if (!student) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    if (!student.isActive) {
      return res.status(403).json({ error: "Your account is inactive. Please contact your administrator." });
    }

    const isMatch = await bcrypt.compare(password, student.passwordHash);
    if (!isMatch) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    // Look up their active assignment from the ExamAssignment collection
    const assignment = await ExamAssignment.findOne({
      studentId: student.studentId,
      status: { $in: ["assigned", "upcoming", "started"] },
    })
      .populate("examId", "title duration isActive")
      .sort({ createdAt: -1 });

    // Compute the effective assignment status based on current time
    let assignedExam = null;
    if (assignment) {
      const now = new Date();
      let computedStatus = assignment.status;
      if (!["completed", "cancelled", "started"].includes(assignment.status)) {
        if (now > assignment.endTime) {
          computedStatus = "expired";
        } else if (now >= assignment.startTime) {
          computedStatus = "active"; // Window is open
        } else {
          computedStatus = "upcoming"; // Not yet started
        }
      }

      assignedExam = {
        assignmentId: assignment._id,
        id: assignment.examId._id,
        title: assignment.examId.title,
        duration: assignment.duration || assignment.examId.duration,
        startTime: assignment.startTime,
        endTime: assignment.endTime,
        status: computedStatus,
      };
    }

    const token = jwt.sign(
      {
        id: student._id,
        studentId: student.studentId,
        name: student.name,
        email: student.email,
        role: "student",
        // Embed minimal assignment info in token for session creation
        assignedExamId: assignedExam?.id || null,
        assignmentId: assignedExam?.assignmentId || null,
        examTitle: assignedExam?.title || null,
        examDuration: assignedExam?.duration || null,
      },
      JWT_SECRET,
      { expiresIn: "4h" }
    );

    return res.json({
      token,
      user: {
        id: student._id,
        studentId: student.studentId,
        name: student.name,
        email: student.email,
        role: "student",
        assignedExam, // null if no active assignment
      },
    });
  } catch (err) {
    console.error("studentLogin error:", err);
    return res.status(500).json({ error: "Server error" });
  }
};
