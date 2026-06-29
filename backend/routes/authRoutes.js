const express = require("express");
const router = express.Router();
const { adminLogin, studentLogin } = require("../controllers/authController");

// POST /api/auth/admin/login
router.post("/admin/login", adminLogin);

// POST /api/auth/student/login
router.post("/student/login", studentLogin);

module.exports = router;
