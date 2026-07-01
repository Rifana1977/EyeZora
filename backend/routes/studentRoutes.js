const express = require("express");
const router = express.Router();
const { verifyToken, requireStudent } = require("../middleware/auth");
const { getExamQuestions, getMyResults } = require("../controllers/studentController");

// Student must be authenticated to fetch exam questions
router.get("/exam/:examId", verifyToken, requireStudent, getExamQuestions);

// Student can view their own published results
router.get("/results", verifyToken, requireStudent, getMyResults);

module.exports = router;