const express = require("express");
const router = express.Router();

const {
  createQuestion,
  getQuestionsByExam,
} = require("../controllers/questionController");

// POST → add question
router.post("/", createQuestion);

// GET → fetch questions for an exam
router.get("/:examId", getQuestionsByExam);

module.exports = router;
