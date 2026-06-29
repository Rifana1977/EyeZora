const express = require("express");
const router = express.Router();
const { verifyToken, requireStudent } = require("../middleware/auth");
const { getExamQuestions } = require("../controllers/studentController");

// Student must be authenticated to fetch exam questions
router.get("/exam/:examId", verifyToken, requireStudent, getExamQuestions);

module.exports = router;