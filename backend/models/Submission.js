const mongoose = require("mongoose");

const submissionSchema = new mongoose.Schema(
  {
    examId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Exam",
      required: true,
    },
    studentId: {
      type: String,
      required: true,
    },
    examSessionId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "ExamSession",
    },
    answers: [Number], // index of selected option per question
    score: {
      type: Number,
      default: 0,
    },
    totalMarks: {
      type: Number,
      default: 0,
    },
    submittedAt: {
      type: Date,
      default: Date.now,
    },
    resultsPublished: {
      type: Boolean,
      default: false,
    },
  },
  { timestamps: true }
);

module.exports = mongoose.model("Submission", submissionSchema);