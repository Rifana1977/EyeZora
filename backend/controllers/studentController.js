const mongoose = require("mongoose");
const Question = require("../models/Question");

/**
 * GET /api/student/exam/:examId
 * Returns questions WITHOUT correctOptionIndex (hidden from students)
 */
exports.getExamQuestions = async (req, res) => {
  try {
    const examId = req.params.examId;

    if (!mongoose.Types.ObjectId.isValid(examId)) {
      return res.status(400).json({ error: "Invalid exam ID" });
    }

    const questions = await Question.find({
      examId: new mongoose.Types.ObjectId(examId),
    })
      .select("-correctOptionIndex")
      .sort({ questionNumber: 1 });

    res.json(questions);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};