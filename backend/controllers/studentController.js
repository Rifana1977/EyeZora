const mongoose = require("mongoose");
const Question = require("../models/Question");
const Submission = require("../models/Submission");
const Exam = require("../models/Exam");

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

/**
 * GET /api/student/results
 * Returns only the student's submissions where resultsPublished = true
 */
exports.getMyResults = async (req, res) => {
  try {
    const { studentId } = req.user;

    const submissions = await Submission.find({
      studentId,
      resultsPublished: true,
    }).sort({ submittedAt: -1 });

    const enriched = await Promise.all(submissions.map(async (sub) => {
      const exam = await Exam.findById(sub.examId).select("title");
      return {
        _id: sub._id,
        examId: sub.examId,
        examTitle: exam?.title || "Unknown Exam",
        score: sub.score,
        totalMarks: sub.totalMarks,
        percentage: sub.totalMarks > 0 ? Math.round((sub.score / sub.totalMarks) * 100) : 0,
        submittedAt: sub.submittedAt,
      };
    }));

    res.json(enriched);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};