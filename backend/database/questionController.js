const Question = require("../models/Question");

/* ---------------- CREATE QUESTION ---------------- */
exports.createQuestion = async (req, res) => {
  try {
    const { examId, questionText, options, correctOptionIndex } = req.body;

    // Basic validation
    if (
      !examId ||
      !questionText ||
      !options ||
      options.length !== 4 ||
      correctOptionIndex === undefined
    ) {
      return res.status(400).json({ message: "Invalid question data" });
    }

    
    const question = await Question.create({
      examId,
      questionText,
      options,
      correctOptionIndex,
    });

    // ✅ return CREATED QUESTION
    res.status(201).json(question);
  } catch (err) {
    console.error("Create Question Error:", err);
    res.status(500).json({ message: "Server error" });
  }
};

/* ---------------- GET QUESTIONS BY EXAM ID ---------------- */
exports.getQuestionsByExam = async (req, res) => {
  try {
    const { examId } = req.params;

    const questions = await Question.find({ examId }).sort({ createdAt: 1 });

    // ✅ ALWAYS return array
    res.status(200).json(questions);
  } catch (err) {
    console.error("Get Questions Error:", err);
    res.status(500).json({ message: "Server error" });
  }
};
