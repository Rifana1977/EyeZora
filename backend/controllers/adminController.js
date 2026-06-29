const Exam = require("../models/Exam");
const Question = require("../models/Question");
const Student = require("../models/Student");
const ExamSession = require("../models/ExamSession");
const ProctoringLog = require("../models/ProctoringLog");
const bcrypt = require("bcryptjs");
const mongoose = require("mongoose");

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