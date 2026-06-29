const mongoose = require("mongoose");

const questionSchema = new mongoose.Schema({
  examId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Exam",
    required: true,
    index: true,
  },
  questionNumber: {
    type: Number,
    required: true,
  },
  questionText: {
    type: String,
    required: true,
  },
  options: [String],
  correctOptionIndex: {
    type: Number,
    required: true,
  },
  marks: {
    type: Number,
    default: 1,
  },
});

module.exports = mongoose.model("Question", questionSchema);