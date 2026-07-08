const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");
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
  bulkDeleteStudents,
  getExamSessions,
  getSessionReport,
  bulkDeleteSessions,
  getDashboardStats,
  bulkImportStudents,
  getResults,
  publishResults,
  unpublishResults,
  bulkDeleteSubmissions,
} = require("../controllers/adminController");

const {
  getAssignments,
  createAssignment,
  bulkCreateAssignment,
  updateAssignment,
  cancelAssignment,
  getStudentAssignment,
  bulkCancelAssignments,
  deleteAssignment,
  bulkDeleteAssignments,
} = require("../controllers/assignmentController");

// Multer config for bulk import (CSV/XLSX)
const importStorage = multer.diskStorage({
  destination: path.join(__dirname, "../temp_uploads/"),
  filename: (req, file, cb) => {
    cb(null, `import_${Date.now()}_${file.originalname}`);
  },
});
const importUpload = multer({
  storage: importStorage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
  fileFilter: (req, file, cb) => {
    const allowed = [".csv", ".xlsx", ".xls"];
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowed.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error("Only CSV and Excel files are allowed"));
    }
  },
});

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
// IMPORTANT: bulk-delete BEFORE /:id so Express doesn't treat "bulk-delete" as an ID
router.delete("/students/bulk-delete", bulkDeleteStudents);
router.put("/students/:id", updateStudent);
router.delete("/students/:id", deleteStudent);
router.post("/students/bulk-import", importUpload.single("file"), bulkImportStudents);

// ── Assignment Routes ───────────────────────────────────
router.get("/assignments", getAssignments);
router.post("/assignments", createAssignment);
router.post("/assignments/bulk", bulkCreateAssignment);
// IMPORTANT: bulk routes BEFORE /:id parameterised routes
router.delete("/assignments/bulk-cancel", bulkCancelAssignments);
router.delete("/assignments", bulkDeleteAssignments);
router.put("/assignments/:id", updateAssignment);
router.delete("/assignments/:id", deleteAssignment);
router.get("/assignments/student/:studentId", getStudentAssignment);

// ── Monitoring Routes ───────────────────────────────────
router.get("/sessions", getExamSessions);
// IMPORTANT: bulk-delete BEFORE /:id/report
router.delete("/sessions/bulk-delete", bulkDeleteSessions);
router.get("/sessions/:id/report", getSessionReport);
router.get("/stats", getDashboardStats);

// ── Results Routes ──────────────────────────────────────
router.get("/results", getResults);
router.post("/results/publish", publishResults);
router.post("/results/unpublish", unpublishResults);
// IMPORTANT: bulk-delete BEFORE any /:id routes
router.delete("/results/bulk-delete", bulkDeleteSubmissions);

module.exports = router;
