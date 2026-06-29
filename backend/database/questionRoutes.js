const express = require("express");
const router = express.Router();
const Question = require("../models/Question");

const {
  createQuestion,
  getQuestionsByExam,
} = require("../controllers/questionController");

// ✅ STUDENT – get questions WITHOUT answers
router.get("/student/:examId", async (req, res) => {
  try {
    const questions = await Question.find(
      { examId: req.params.examId },
      { correctOptionIndex: 0 } // hide correct answer
    );
    res.json(questions);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ✅ ADMIN – create question
router.post("/", createQuestion);

// ✅ ADMIN – get full questions (with answers)
router.get("/:examId", getQuestionsByExam);

module.exports = router;
