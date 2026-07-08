const Exam = require("../models/Exam");
const Question = require("../models/Question");
const Student = require("../models/Student");
const ExamSession = require("../models/ExamSession");
const ProctoringLog = require("../models/ProctoringLog");
const Submission = require("../models/Submission");
const bcrypt = require("bcryptjs");
const crypto = require("crypto");
const nodemailer = require("nodemailer");
const { parse: csvParse } = require("csv-parse/sync");
const XLSX = require("xlsx");
const mongoose = require("mongoose");
const transporter = require("../config/mailer");

// ─── Exam Management ───────────────────────────────────────────────────────────

/**
 * POST /api/admin/exam
 * Create a new exam
 */
exports.createExam = async (req, res) => {
  try {
    const { title, duration } = req.body;
    if (!title) return res.status(400).json({ error: "Title is required" });

    const exam = await Exam.create({
      title,
      duration: duration || 60,
      createdBy: req.user?.id || "admin",
    });
    res.status(201).json(exam);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

/**
 * GET /api/admin/exams
 * List all exams (for dropdown selections)
 */
exports.getAllExams = async (req, res) => {
  try {
    const exams = await Exam.find().sort({ createdAt: -1 });
    res.json(exams);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

/**
 * PUT /api/admin/exam/:id
 * Update an existing exam
 */
exports.updateExam = async (req, res) => {
  try {
    const { id } = req.params;
    const { title, duration } = req.body;
    if (!title) return res.status(400).json({ error: "Title is required" });

    const exam = await Exam.findByIdAndUpdate(
      id,
      { title, duration: duration || 60 },
      { new: true, runValidators: true }
    );

    if (!exam) return res.status(404).json({ error: "Exam not found" });
    res.json(exam);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

/**
 * DELETE /api/admin/exam/:id
 * Delete an exam and all its questions
 */
exports.deleteExam = async (req, res) => {
  try {
    const { id } = req.params;
    const exam = await Exam.findById(id);
    if (!exam) return res.status(404).json({ error: "Exam not found" });

    await exam.deleteOne();
    // Delete all questions associated with this exam
    await Question.deleteMany({ examId: id });

    res.json({ message: "Exam and all its questions deleted successfully" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// ─── Question Management ───────────────────────────────────────────────────────

/**
 * POST /api/admin/question
 * Add a question to a test
 */
exports.addQuestion = async (req, res) => {
  try {
    const { examId, questionText, options, correctOptionIndex, marks } = req.body;

    if (!examId) return res.status(400).json({ error: "examId is required" });
    if (!questionText) return res.status(400).json({ error: "questionText is required" });
    if (correctOptionIndex === undefined)
      return res.status(400).json({ error: "correctOptionIndex is required" });

    // Auto-assign questionNumber
    const count = await Question.countDocuments({ examId });

    const question = await Question.create({
      examId,
      questionNumber: count + 1,
      questionText,
      options: options || [],
      correctOptionIndex,
      marks: marks || 1,
    });

    res.status(201).json(question);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

/**
 * GET /api/admin/test/:testId/questions
 * Get all questions for a test with pagination
 */
exports.getTestWithQuestions = async (req, res) => {
  try {
    const { testId } = req.params;
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 20;
    const skip = (page - 1) * limit;

    if (!mongoose.Types.ObjectId.isValid(testId)) {
      return res.status(400).json({ error: "Invalid Test ID format" });
    }

    const exam = await Exam.findById(testId);
    if (!exam) return res.status(404).json({ error: "Test not found" });

    const total = await Question.countDocuments({ examId: testId });
    const questions = await Question.find({ examId: testId })
      .sort({ questionNumber: 1 })
      .skip(skip)
      .limit(limit);

    res.json({
      exam,
      questions,
      pagination: {
        total,
        page,
        limit,
        totalPages: Math.ceil(total / limit),
      },
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

/**
 * PUT /api/admin/question/:id
 * Update a question
 */
exports.updateQuestion = async (req, res) => {
  try {
    const { id } = req.params;
    const updates = req.body;

    // Prevent changing examId
    delete updates.examId;

    const question = await Question.findByIdAndUpdate(id, updates, {
      new: true,
      runValidators: true,
    });

    if (!question) return res.status(404).json({ error: "Question not found" });
    res.json(question);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

/**
 * DELETE /api/admin/question/:id
 * Delete a question and renumber remaining
 */
exports.deleteQuestion = async (req, res) => {
  try {
    const { id } = req.params;
    const question = await Question.findById(id);
    if (!question) return res.status(404).json({ error: "Question not found" });

    const { examId, questionNumber } = question;
    await question.deleteOne();

    // Renumber questions after the deleted one
    await Question.updateMany(
      { examId, questionNumber: { $gt: questionNumber } },
      { $inc: { questionNumber: -1 } }
    );

    res.json({ message: "Question deleted successfully" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// ─── Student Management ────────────────────────────────────────────────────────

/**
 * GET /api/admin/students
 * List all pre-registered students (no exam population — use ExamAssignment)
 */
exports.getStudents = async (req, res) => {
  try {
    const students = await Student.find()
      .select("-passwordHash")
      .sort({ createdAt: -1 });
    res.json(students);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

/**
 * POST /api/admin/students
 * Register a student — exam assignment is optional and done separately
 */
exports.registerStudent = async (req, res) => {
  try {
    const { studentId, name, email, password, isActive } = req.body;

    if (!studentId || !name || !email || !password) {
      return res.status(400).json({ error: "studentId, name, email, and password are required" });
    }

    const exists = await Student.findOne({
      $or: [{ studentId }, { email: email.toLowerCase() }],
    });
    if (exists) {
      return res.status(409).json({ error: "Student ID or email already exists" });
    }

    const passwordHash = await bcrypt.hash(password, 12);
    const student = await Student.create({
      studentId,
      name,
      email: email.toLowerCase(),
      passwordHash,
      isActive: isActive !== undefined ? isActive : true,
    });

    // Return without passwordHash
    const { passwordHash: _omit, ...studentObj } = student.toObject();
    res.status(201).json(studentObj);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

/**
 * PUT /api/admin/students/:id
 * Update student details (name, email, status)
 */
exports.updateStudent = async (req, res) => {
  try {
    const { name, email, isActive, password } = req.body;
    const updates = {};
    if (name) updates.name = name;
    if (email) updates.email = email.toLowerCase();
    if (isActive !== undefined) updates.isActive = isActive;
    if (password) {
      updates.passwordHash = await bcrypt.hash(password, 12);
    }

    const student = await Student.findByIdAndUpdate(req.params.id, updates, {
      new: true,
      runValidators: true,
    }).select("-passwordHash");

    if (!student) return res.status(404).json({ error: "Student not found" });
    res.json(student);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

/**
 * DELETE /api/admin/students/:id
 * Remove a student
 */
exports.deleteStudent = async (req, res) => {
  try {
    const student = await Student.findByIdAndDelete(req.params.id);
    if (!student) return res.status(404).json({ error: "Student not found" });
    res.json({ message: "Student deleted" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

/**
 * DELETE /api/admin/students/bulk-delete
 * Body: { ids: string[] }
 * Hard-delete multiple students at once.
 */
exports.bulkDeleteStudents = async (req, res) => {
  try {
    const { ids } = req.body;
    if (!ids || !Array.isArray(ids) || ids.length === 0) {
      return res.status(400).json({ error: "ids array is required" });
    }
    const result = await Student.deleteMany({ _id: { $in: ids } });
    res.json({ deleted: result.deletedCount, requested: ids.length });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// ─── Admin Monitoring Dashboard ────────────────────────────────────────────────

/**
 * GET /api/admin/sessions
 * Get all exam sessions for monitoring
 */
exports.getExamSessions = async (req, res) => {
  try {
    const sessions = await ExamSession.find()
      .sort({ createdAt: -1 });
    res.json(sessions);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

/**
 * GET /api/admin/sessions/:id/report
 * Get detailed report for one session
 */
exports.getSessionReport = async (req, res) => {
  try {
    const session = await ExamSession.findById(req.params.id);
    if (!session) return res.status(404).json({ error: "Session not found" });

    const logs = await ProctoringLog.find({ sessionId: session._id }).sort({
      timestamp: 1,
    });

    res.json({ session, logs });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

/**
 * DELETE /api/admin/sessions/bulk-delete
 * Body: { ids: string[] }
 * Hard-delete multiple exam session records and their proctoring logs.
 */
exports.bulkDeleteSessions = async (req, res) => {
  try {
    const { ids } = req.body;
    if (!ids || !Array.isArray(ids) || ids.length === 0) {
      return res.status(400).json({ error: "ids array is required" });
    }
    // Also remove associated proctoring logs
    await ProctoringLog.deleteMany({ sessionId: { $in: ids } });
    const result = await ExamSession.deleteMany({ _id: { $in: ids } });
    res.json({ deleted: result.deletedCount, requested: ids.length });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

/**
 * GET /api/admin/stats
 * Overview stats for admin dashboard
 */
exports.getDashboardStats = async (req, res) => {
  try {
    const totalExams = await Exam.countDocuments();
    const totalStudents = await Student.countDocuments();
    const completedSessions = await ExamSession.countDocuments({
      status: { $in: ["completed", "flagged"] },
    });
    const flaggedSessions = await ExamSession.countDocuments({
      riskLevel: "High",
    });
    const recentSessions = await ExamSession.find()
      .sort({ createdAt: -1 })
      .limit(5);

    res.json({
      totalExams,
      totalStudents,
      completedSessions,
      flaggedSessions,
      recentSessions,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// ─── Bulk Student Import ───────────────────────────────────────────────────────

/**
 * POST /api/admin/students/bulk-import
 * Accepts a CSV or XLSX file via multipart form.
 * Expected columns: studentId, name, email (case-insensitive headers)
 * Creates accounts with secure random temporary passwords.
 * Optionally emails credentials to each student.
 * Returns a credentials array for CSV download.
 */
exports.bulkImportStudents = async (req, res) => {
  try {
    const file = req.file;
    if (!file) {
      return res.status(400).json({ error: "Please upload a CSV or XLSX file" });
    }

    let rows = [];

    if (file.mimetype === "text/csv" || file.originalname.endsWith(".csv")) {
      // Parse CSV
      const content = require("fs").readFileSync(file.path, "utf-8");
      rows = csvParse(content, {
        columns: true,
        skip_empty_lines: true,
        trim: true,
      });
    } else {
      // Parse XLSX
      const workbook = XLSX.readFile(file.path);
      const sheetName = workbook.SheetNames[0];
      const sheet = workbook.Sheets[sheetName];
      rows = XLSX.utils.sheet_to_json(sheet, { defval: "" });
    }

    // Clean up temp file
    const fs = require("fs");
    if (fs.existsSync(file.path)) fs.unlinkSync(file.path);

    if (rows.length === 0) {
      return res.status(400).json({ error: "The uploaded file has no data rows" });
    }

    const results = [];
    const errors = [];
    const sendEmails = req.body.sendEmails === "true";

    // Using centralized email transporter


    for (const row of rows) {
      // Normalize column names (case-insensitive)
      const normalizedRow = {};
      for (const [k, v] of Object.entries(row)) {
        normalizedRow[k.toLowerCase().replace(/\s+/g, "")] = String(v).trim();
      }

      const studentId = normalizedRow["studentid"] || normalizedRow["student_id"] || normalizedRow["id"] || "";
      const name = normalizedRow["name"] || normalizedRow["studentname"] || normalizedRow["student_name"] || "";
      const email = (normalizedRow["email"] || "").toLowerCase();

      if (!studentId || !name || !email) {
        errors.push({ row: JSON.stringify(row), error: "Missing required fields (studentId, name, email)" });
        continue;
      }

      // Check if already exists
      const existing = await Student.findOne({
        $or: [{ studentId }, { email }],
      });
      if (existing) {
        errors.push({ studentId, email, error: "Student ID or email already exists" });
        continue;
      }

      // Generate a secure random temporary password
      const tempPassword = crypto.randomBytes(6).toString("base64").replace(/[^a-zA-Z0-9]/g, "").slice(0, 10);
      const passwordHash = await bcrypt.hash(tempPassword, 12);

      try {
        await Student.create({
          studentId,
          name,
          email,
          passwordHash,
          isActive: true,
          isTemporaryPassword: true,
        });

        results.push({ studentId, name, email, tempPassword });

        // Optionally send credentials email
        if (sendEmails && process.env.EMAIL_FROM) {
          const frontendUrl = process.env.FRONTEND_URL || "http://localhost:3000";
          try {
            await transporter.sendMail({
              from: `"EyeZora Exam System" <${process.env.EMAIL_FROM}>`,
              to: email,
              subject: "Your EyeZora Exam Account Credentials",
              html: `
                <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;padding:24px;background:#f8f7ff;border-radius:12px;">
                  <div style="text-align:center;margin-bottom:24px;">
                    <h1 style="color:#7c3aed;font-size:26px;letter-spacing:2px;">👁 EYEZORA</h1>
                    <p style="color:#64748b;font-size:13px;">AI-Powered Examination System</p>
                  </div>
                  <div style="background:#fff;padding:28px;border-radius:10px;border:1px solid rgba(124,58,237,0.15);">
                    <h2 style="color:#1e1b4b;font-size:18px;margin-bottom:12px;">Your Exam Account is Ready</h2>
                    <p style="color:#374151;font-size:14px;line-height:1.7;">Hi <strong>${name}</strong>,</p>
                    <p style="color:#374151;font-size:14px;line-height:1.7;">
                      Your EyeZora examination account has been created. Use the credentials below to log in.
                      <strong>You will be required to change your password on first login.</strong>
                    </p>
                    <div style="background:#f3f0ff;padding:18px;border-radius:8px;margin:20px 0;border:1px solid rgba(124,58,237,0.2);">
                      <table style="width:100%;font-size:14px;">
                        <tr><td style="color:#64748b;padding:4px 0;width:40%;">Student ID:</td><td style="font-weight:700;color:#1e1b4b;">${studentId}</td></tr>
                        <tr><td style="color:#64748b;padding:4px 0;">Email:</td><td style="font-weight:700;color:#1e1b4b;">${email}</td></tr>
                        <tr><td style="color:#64748b;padding:4px 0;">Temporary Password:</td><td style="font-weight:700;color:#7c3aed;font-family:monospace;font-size:15px;">${tempPassword}</td></tr>
                      </table>
                    </div>
                    <div style="text-align:center;margin:24px 0;">
                      <a href="${frontendUrl}/student/login"
                         style="background:linear-gradient(135deg,#7c3aed,#5b21b6);color:#fff;padding:12px 28px;
                                border-radius:10px;text-decoration:none;font-weight:700;font-size:14px;display:inline-block;">
                        Login to EyeZora
                      </a>
                    </div>
                    <p style="color:#9ca3af;font-size:12px;margin-top:16px;line-height:1.6;">
                      Please change your password immediately after your first login.
                      Do not share your credentials with anyone.
                    </p>
                  </div>
                </div>
              `,
            });
          } catch (emailErr) {
            console.error(`Email failed for ${email}:`, emailErr.message);
          }
        }
      } catch (createErr) {
        errors.push({ studentId, email, error: createErr.message });
      }
    }

    res.status(201).json({
      created: results.length,
      failed: errors.length,
      credentials: results,  // admin can download this as CSV
      errors,
    });
  } catch (err) {
    console.error("bulkImportStudents error:", err);
    res.status(500).json({ error: err.message });
  }
};

// ─── Result Publication ────────────────────────────────────────────────────────

/**
 * GET /api/admin/results
 * List all submissions with publish status (optionally filter by examId)
 */
exports.getResults = async (req, res) => {
  try {
    const { examId } = req.query;
    const filter = {};
    if (examId) filter.examId = examId;

    const submissions = await Submission.find(filter)
      .sort({ submittedAt: -1 });

    // Enrich with student and exam info
    const enriched = await Promise.all(submissions.map(async (sub) => {
      const student = await Student.findOne({ studentId: sub.studentId }).select("name studentId email");
      const exam = await Exam.findById(sub.examId).select("title");
      return {
        _id: sub._id,
        studentId: sub.studentId,
        studentName: student?.name || sub.studentId,
        studentEmail: student?.email || "",
        examId: sub.examId,
        examTitle: exam?.title || "Unknown",
        score: sub.score,
        totalMarks: sub.totalMarks,
        percentage: sub.totalMarks > 0 ? Math.round((sub.score / sub.totalMarks) * 100) : 0,
        resultsPublished: sub.resultsPublished,
        submittedAt: sub.submittedAt,
      };
    }));

    res.json(enriched);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

/**
 * POST /api/admin/results/publish
 * Body: { examId } — publishes all submissions for the given exam
 * OR  { submissionIds: [] } — publish specific submissions
 */
exports.publishResults = async (req, res) => {
  try {
    const { examId, submissionIds } = req.body;

    if (!examId && (!submissionIds || submissionIds.length === 0)) {
      return res.status(400).json({ error: "examId or submissionIds is required" });
    }

    let filter = {};
    if (submissionIds && submissionIds.length > 0) {
      filter._id = { $in: submissionIds };
    } else if (examId) {
      filter.examId = examId;
    }

    // Find all matching submissions
    const submissions = await Submission.find(filter);

    if (submissions.length === 0) {
      return res.status(404).json({ error: "No submissions found to publish" });
    }

    let processedCount = 0;
    let sentCount = 0;
    let failedCount = 0;

    for (const sub of submissions) {
      processedCount++;
      const studentId = sub.studentId;
      console.log(`✓ Publishing result for Student ID ${studentId}...`);

      try {
        // Fetch student
        const student = await Student.findOne({ studentId });
        if (!student) {
          throw new Error(`Student ${studentId} not found in database`);
        }

        // Fetch exam
        const exam = await Exam.findById(sub.examId);
        const examTitle = exam ? exam.title : "Examination";

        // Update database status
        sub.resultsPublished = true;
        await sub.save();

        console.log(`✓ Preparing email...`);
        const percentage = sub.totalMarks > 0 ? Math.round((sub.score / sub.totalMarks) * 100) : 0;
        const resultPayload = {
          examTitle,
          score: sub.score,
          totalMarks: sub.totalMarks,
          percentage,
          publishedAt: new Date(),
        };

        console.log(`✓ Sending email...`);
        await transporter.sendResultPublishedEmail(student, resultPayload);
        console.log(`✓ Email sent successfully.`);
        sentCount++;
      } catch (err) {
        console.log(`✗ Email failed.`);
        console.log(`Reason: ${err.message}`);
        failedCount++;
      }
    }

    const summary = `Published Results Summary\n-------------------------\nStudents Processed: ${processedCount}\nEmails Sent: ${sentCount}\nEmails Failed: ${failedCount}`;

    return res.json({
      message: summary,
      processed: processedCount,
      sent: sentCount,
      failed: failedCount,
    });
  } catch (err) {
    console.error("publishResults error:", err);
    res.status(500).json({ error: err.message });
  }
};

/**
 * POST /api/admin/results/unpublish
 * Body: { examId } or { submissionIds: [] }
 */
exports.unpublishResults = async (req, res) => {
  try {
    const { examId, submissionIds } = req.body;
    const filter = {};
    if (submissionIds && submissionIds.length > 0) {
      filter._id = { $in: submissionIds };
    } else if (examId) {
      filter.examId = examId;
    } else {
      return res.status(400).json({ error: "examId or submissionIds is required" });
    }

    const result = await Submission.updateMany(filter, { resultsPublished: false });

    res.json({
      message: `Results unpublished for ${result.modifiedCount} submission(s)`,
      modifiedCount: result.modifiedCount,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

/**
 * DELETE /api/admin/results/bulk-delete
 * Body: { ids: string[] }
 * Hard-delete multiple submission (result) records.
 */
exports.bulkDeleteSubmissions = async (req, res) => {
  try {
    const { ids } = req.body;
    if (!ids || !Array.isArray(ids) || ids.length === 0) {
      return res.status(400).json({ error: "ids array is required" });
    }
    const result = await Submission.deleteMany({ _id: { $in: ids } });
    res.json({ deleted: result.deletedCount, requested: ids.length });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};