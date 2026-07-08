const express = require("express");
const router = express.Router();
const { verifyToken, requireStudent } = require("../middleware/auth");
const {
  adminLogin,
  studentLogin,
  forgotPassword,
  resetPassword,
  changePassword,
} = require("../controllers/authController");

// POST /api/auth/admin/login
router.post("/admin/login", adminLogin);

// POST /api/auth/student/login
router.post("/student/login", studentLogin);

// ── Forgot Password ───────────────────────────────────────────────────────────
// Primary route (as specified in the new API design)
router.post("/forgot-password", forgotPassword);

// Backward-compatible alias (keeps existing integrations working)
router.post("/student/forgot-password", forgotPassword);

// ── Reset Password ────────────────────────────────────────────────────────────
// Primary route (as specified in the new API design)
router.post("/reset-password", resetPassword);

// Backward-compatible alias
router.post("/student/reset-password", resetPassword);

// ── Change Password (authenticated) ──────────────────────────────────────────
router.post("/student/change-password", verifyToken, requireStudent, changePassword);

module.exports = router;

