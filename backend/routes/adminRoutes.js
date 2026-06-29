const express = require("express");
const router = express.Router();
const { verifyToken, requireAdmin } = require("../middleware/auth");

const {
  createExam,
  getAllExams,
  updateExam,
  deleteExam,
  addQuestion,
  getTestWithQuestions,
  updateQuestion,
  deleteQuestion,
  getStudents,
  registerStudent,
  updateStudent,
  deleteStudent,
  getExamSessions,
  getSessionReport,
  getDashboardStats,
} = require("../controllers/adminController");

const {
  getAssignments,
  createAssignment,
  updateAssignment,
  cancelAssignment,
  getStudentAssignment,
} = require("../controllers/assignmentController");

// All admin routes require authentication
router.use(verifyToken, requireAdmin);

// ── Exam Routes ─────────────────────────────────────────
router.post("/exam", createExam);
router.get("/exams", getAllExams);
router.put("/exam/:id", updateExam);
router.delete("/exam/:id", deleteExam);

// ── Question Routes ─────────────────────────────────────
router.post("/question", addQuestion);
router.get("/test/:testId/questions", getTestWithQuestions);
router.put("/question/:id", updateQuestion);
router.delete("/question/:id", deleteQuestion);

// ── Student Routes ──────────────────────────────────────
router.get("/students", getStudents);
router.post("/students", registerStudent);
router.put("/students/:id", updateStudent);
router.delete("/students/:id", deleteStudent);

// ── Assignment Routes ───────────────────────────────────
router.get("/assignments", getAssignments);
router.post("/assignments", createAssignment);
router.put("/assignments/:id", updateAssignment);
router.delete("/assignments/:id", cancelAssignment);
router.get("/assignments/student/:studentId", getStudentAssignment);

// ── Monitoring Routes ───────────────────────────────────
router.get("/sessions", getExamSessions);
router.get("/sessions/:id/report", getSessionReport);
router.get("/stats", getDashboardStats);

module.exports = router;