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

// POST /api/auth/student/forgot-password
router.post("/student/forgot-password", forgotPassword);

// POST /api/auth/student/reset-password
router.post("/student/reset-password", resetPassword);

// POST /api/auth/student/change-password (must be logged in)
router.post("/student/change-password", verifyToken, requireStudent, changePassword);

module.exports = router;

