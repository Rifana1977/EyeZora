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
 * POST /api/auth/forgot-password  (also aliased at /api/auth/student/forgot-password)
 * Body: { studentId, email }
 *
 * Security:
 *  - Both studentId AND email must belong to the same student document.
 *  - Always returns the same generic message to prevent enumeration attacks.
 *  - Token is generated with crypto.randomBytes(32); only its SHA-256 hash is stored.
 *  - Token expires in 15 minutes.
 */
exports.forgotPassword = async (req, res) => {
  const GENERIC_RESPONSE = {
    message: "If the provided details are valid, a password reset link has been sent.",
  };

  try {
    const { studentId, email } = req.body;

    // Require both fields — never reveal which one is wrong/missing
    if (!studentId || !email) {
      return res.json(GENERIC_RESPONSE);
    }

    // Both studentId AND email must match the SAME document
    const student = await Student.findOne({
      studentId: studentId.trim(),
      email: email.toLowerCase().trim(),
    });

    // No match — return generic response (prevents enumeration)
    if (!student) {
      return res.json(GENERIC_RESPONSE);
    }

    // Generate a cryptographically secure raw token
    const rawToken = crypto.randomBytes(32).toString("hex");

    // Store only the SHA-256 hash — raw token is never persisted
    const hashedToken = crypto.createHash("sha256").update(rawToken).digest("hex");

    // Token expires in exactly 15 minutes
    const expiry = new Date(Date.now() + 15 * 60 * 1000);

    await Student.findByIdAndUpdate(student._id, {
      passwordResetToken: hashedToken,
      passwordResetExpires: expiry,
    });

    // Reset link embeds the RAW token in the path (not a query param)
    const frontendUrl = process.env.FRONTEND_URL || "http://localhost:3000";
    const resetUrl = `${frontendUrl}/reset-password/${rawToken}`;

    try {
      await transporter.sendMail({
        from: `"EyeZora Exam System" <${process.env.EMAIL_FROM}>`,
        to: student.email,
        subject: "EyeZora Password Reset Request",
        html: `
          <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;padding:0;">
            <!-- Header -->
            <div style="background:linear-gradient(135deg,#7c3aed,#5b21b6);padding:32px 24px;border-radius:12px 12px 0 0;text-align:center;">
              <h1 style="color:#fff;font-size:28px;letter-spacing:3px;margin:0 0 4px;">👁 EYEZORA</h1>
              <p style="color:rgba(255,255,255,0.75);font-size:13px;margin:0;">AI-Powered Examination System</p>
            </div>

            <!-- Body -->
            <div style="background:#fff;padding:36px 32px;border-left:1px solid #e5e7eb;border-right:1px solid #e5e7eb;">
              <h2 style="color:#1e1b4b;font-size:20px;font-weight:700;margin:0 0 16px;">Password Reset Request</h2>

              <p style="color:#374151;font-size:14px;line-height:1.75;margin:0 0 8px;">
                Hello <strong>${student.name}</strong>,
              </p>
              <p style="color:#374151;font-size:14px;line-height:1.75;margin:0 0 24px;">
                A request was made to reset the password for your EyeZora account.
              </p>

              <!-- Student ID badge -->
              <div style="background:#f3f0ff;border:1px solid rgba(124,58,237,0.2);border-radius:8px;padding:12px 16px;margin-bottom:28px;">
                <p style="color:#6b7280;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin:0 0 4px;">Student ID</p>
                <p style="color:#1e1b4b;font-size:15px;font-weight:700;margin:0;">${student.studentId}</p>
              </div>

              <p style="color:#374151;font-size:14px;line-height:1.75;margin:0 0 24px;">
                Click the button below to reset your password.
              </p>

              <!-- CTA Button -->
              <div style="text-align:center;margin:0 0 28px;">
                <a href="${resetUrl}"
                   style="background:linear-gradient(135deg,#7c3aed,#5b21b6);color:#fff;padding:16px 40px;
                          border-radius:10px;text-decoration:none;font-weight:700;font-size:15px;
                          display:inline-block;box-shadow:0 4px 15px rgba(124,58,237,0.4);letter-spacing:0.3px;">
                  Reset Password
                </a>
              </div>

              <p style="color:#6b7280;font-size:13px;line-height:1.75;margin:0 0 16px;">
                This link expires in <strong>15 minutes</strong>.
              </p>

              <p style="color:#6b7280;font-size:13px;line-height:1.75;margin:0;">
                If you did not request this reset, simply ignore this email.
                Your current password will continue to work unless you choose to change it.
              </p>
            </div>

            <!-- Footer -->
            <div style="background:#f9fafb;padding:16px 24px;border-radius:0 0 12px 12px;
                        border:1px solid #e5e7eb;border-top:none;text-align:center;">
              <p style="color:#9ca3af;font-size:11px;margin:0;">
                This is an automated message from EyeZora. Please do not reply to this email.
              </p>
            </div>
          </div>
        `,
      });
    } catch (emailErr) {
      // Log but do not expose email errors to the caller
      console.error("[forgotPassword] Email send error:", emailErr.message);
    }

    return res.json(GENERIC_RESPONSE);
  } catch (err) {
    console.error("forgotPassword error:", err);
    return res.status(500).json({ error: "Server error" });
  }
};

// ─── Reset Password ────────────────────────────────────────────────────────────

/**
 * POST /api/auth/reset-password  (also aliased at /api/auth/student/reset-password)
 * Body: { token, password }
 *
 * Security:
 *  - Looks up the student by the SHA-256 hash of the supplied raw token only.
 *  - Email is NOT required in the body (token lookup is sufficient).
 *  - Validates token expiry.
 *  - Token is single-use: cleared immediately after a successful reset.
 */
exports.resetPassword = async (req, res) => {
  try {
    const { token, password } = req.body;

    if (!token || !password) {
      return res.status(400).json({ error: "Token and password are required" });
    }
    if (password.length < 8) {
      return res.status(400).json({ error: "Password must be at least 8 characters" });
    }

    // Hash the raw token to compare against the stored hash
    const hashedToken = crypto.createHash("sha256").update(token).digest("hex");

    const student = await Student.findOne({
      passwordResetToken: hashedToken,
      passwordResetExpires: { $gt: new Date() }, // must not be expired
    });

    if (!student) {
      return res
        .status(400)
        .json({ error: "Invalid or expired reset link. Please request a new one." });
    }

    // Hash the new password with bcrypt (cost factor 12)
    const passwordHash = await bcrypt.hash(password, 12);

    // Update password and immediately invalidate the token (single-use)
    await Student.findByIdAndUpdate(student._id, {
      passwordHash,
      passwordResetToken: null,
      passwordResetExpires: null,
      isTemporaryPassword: false,
    });

    return res.json({
      message: "Password reset successfully. You can now log in with your new password.",
    });
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


