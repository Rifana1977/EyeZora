const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const crypto = require("crypto");
const nodemailer = require("nodemailer");
const Admin = require("../models/Admin");
const Student = require("../models/Student");
const ExamAssignment = require("../models/ExamAssignment");
const { JWT_SECRET } = require("../middleware/auth");

const transporter = require("../config/mailer");

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
 * Returns isTemporaryPassword flag so frontend can redirect to change-password.
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
        assignedExamId: assignedExam?.id || null,
        assignmentId: assignedExam?.assignmentId || null,
        examTitle: assignedExam?.title || null,
        examDuration: assignedExam?.duration || null,
        isTemporaryPassword: student.isTemporaryPassword,
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
        isTemporaryPassword: student.isTemporaryPassword,
        assignedExam,
      },
    });
  } catch (err) {
    console.error("studentLogin error:", err);
    return res.status(500).json({ error: "Server error" });
  }
};

// ─── Forgot Password ───────────────────────────────────────────────────────────

/**
 * POST /api/auth/student/forgot-password
 * Body: { email }
 */
exports.forgotPassword = async (req, res) => {
  try {
    const { email } = req.body;
    if (!email) return res.status(400).json({ error: "Email is required" });

    const student = await Student.findOne({ email: email.toLowerCase() });

    // Always respond with success to prevent email enumeration
    if (!student) {
      return res.json({ message: "If that email is registered, a reset link has been sent." });
    }

    const rawToken = crypto.randomBytes(32).toString("hex");
    const hashedToken = crypto.createHash("sha256").update(rawToken).digest("hex");
    const expiryHours = Number(process.env.RESET_TOKEN_EXPIRY_HOURS) || 2;
    const expiry = new Date(Date.now() + expiryHours * 60 * 60 * 1000);

    await Student.findByIdAndUpdate(student._id, {
      passwordResetToken: hashedToken,
      passwordResetExpires: expiry,
    });

    const frontendUrl = process.env.FRONTEND_URL || "http://localhost:3000";
    const resetUrl = `${frontendUrl}/student/reset-password?token=${rawToken}&email=${encodeURIComponent(student.email)}`;

    try {
      await transporter.sendMail({
        from: `"EyeZora Exam System" <${process.env.EMAIL_FROM}>`,
        to: student.email,
        subject: "EyeZora — Password Reset Request",
        html: `
          <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;padding:24px;background:#f8f7ff;border-radius:12px;">
            <div style="text-align:center;margin-bottom:28px;">
              <h1 style="color:#7c3aed;font-size:28px;letter-spacing:2px;">👁 EYEZORA</h1>
              <p style="color:#64748b;font-size:13px;">AI-Powered Examination System</p>
            </div>
            <div style="background:#fff;padding:28px;border-radius:10px;border:1px solid rgba(124,58,237,0.15);">
              <h2 style="color:#1e1b4b;font-size:20px;margin-bottom:12px;">Password Reset Request</h2>
              <p style="color:#374151;font-size:14px;line-height:1.7;">Hi <strong>${student.name}</strong>,</p>
              <p style="color:#374151;font-size:14px;line-height:1.7;">
                We received a request to reset your EyeZora account password.
                Click the button below to create a new password.
              </p>
              <div style="text-align:center;margin:28px 0;">
                <a href="${resetUrl}"
                   style="background:linear-gradient(135deg,#7c3aed,#5b21b6);color:#fff;padding:14px 32px;
                          border-radius:10px;text-decoration:none;font-weight:700;font-size:15px;display:inline-block;">
                  Reset My Password
                </a>
              </div>
              <p style="color:#64748b;font-size:13px;line-height:1.7;">
                This link expires in <strong>${expiryHours} hours</strong>.
                If you did not request this, you can safely ignore this email.
              </p>
              <p style="color:#9ca3af;font-size:12px;margin-top:20px;border-top:1px solid #f3f4f6;padding-top:16px;">
                Student ID: ${student.studentId}
              </p>
            </div>
          </div>
        `,
      });
    } catch (emailErr) {
      console.error("Email send error:", emailErr.message);
    }

    return res.json({ message: "If that email is registered, a reset link has been sent." });
  } catch (err) {
    console.error("forgotPassword error:", err);
    return res.status(500).json({ error: "Server error" });
  }
};

// ─── Reset Password ────────────────────────────────────────────────────────────

/**
 * POST /api/auth/student/reset-password
 * Body: { token, email, newPassword }
 */
exports.resetPassword = async (req, res) => {
  try {
    const { token, email, newPassword } = req.body;
    if (!token || !email || !newPassword) {
      return res.status(400).json({ error: "Token, email, and newPassword are required" });
    }
    if (newPassword.length < 8) {
      return res.status(400).json({ error: "Password must be at least 8 characters" });
    }

    const hashedToken = crypto.createHash("sha256").update(token).digest("hex");

    const student = await Student.findOne({
      email: email.toLowerCase(),
      passwordResetToken: hashedToken,
      passwordResetExpires: { $gt: new Date() },
    });

    if (!student) {
      return res.status(400).json({ error: "Invalid or expired reset link. Please request a new one." });
    }

    const passwordHash = await bcrypt.hash(newPassword, 12);

    await Student.findByIdAndUpdate(student._id, {
      passwordHash,
      passwordResetToken: null,
      passwordResetExpires: null,
      isTemporaryPassword: false,
    });

    return res.json({ message: "Password reset successfully. You can now log in with your new password." });
  } catch (err) {
    console.error("resetPassword error:", err);
    return res.status(500).json({ error: "Server error" });
  }
};

// ─── Change Password (First Login / Authenticated) ─────────────────────────────

/**
 * POST /api/auth/student/change-password
 * Authenticated — student must be logged in.
 * Body: { currentPassword, newPassword }
 */
exports.changePassword = async (req, res) => {
  try {
    const { currentPassword, newPassword } = req.body;
    const studentId = req.user.id;

    if (!currentPassword || !newPassword) {
      return res.status(400).json({ error: "currentPassword and newPassword are required" });
    }
    if (newPassword.length < 8) {
      return res.status(400).json({ error: "New password must be at least 8 characters" });
    }

    const student = await Student.findById(studentId);
    if (!student) return res.status(404).json({ error: "Student not found" });

    const isMatch = await bcrypt.compare(currentPassword, student.passwordHash);
    if (!isMatch) {
      return res.status(401).json({ error: "Current password is incorrect" });
    }
    if (currentPassword === newPassword) {
      return res.status(400).json({ error: "New password must be different from your current password" });
    }

    const passwordHash = await bcrypt.hash(newPassword, 12);
    await Student.findByIdAndUpdate(studentId, {
      passwordHash,
      isTemporaryPassword: false,
    });

    return res.json({ message: "Password changed successfully." });
  } catch (err) {
    console.error("changePassword error:", err);
    return res.status(500).json({ error: "Server error" });
  }
};


