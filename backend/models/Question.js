const mongoose = require("mongoose");

const questionSchema = new mongoose.Schema({
  examId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Exam",
    required: true
  },
  questionText: {
    type: String,
    required: true
  },
  options: {
    type: [Object], // or [String] depending on your design
    required: true
  },
  correctOptionIndex: {
    type: Number,
    required: true
  }
});

module.exports = mongoose.model("Question", questionSchema);
