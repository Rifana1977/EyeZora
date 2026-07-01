const mongoose = require("mongoose");

/**
 * Student model — students are pre-registered by admin before the exam.
 * They cannot self-register.
 * Exam assignment is now handled by the ExamAssignment collection.
 */
const studentSchema = new mongoose.Schema(
  {
    studentId: {
      type: String,
      required: true,
      unique: true,
      trim: true,
      // e.g. "STU001"
    },
    name: {
      type: String,
      required: true,
      trim: true,
    },
    email: {
      type: String,
      required: true,
      unique: true,
      lowercase: true,
      trim: true,
    },
    passwordHash: {
      type: String,
      required: true,
    },
    isTemporaryPassword: {
      type: Boolean,
      default: false,
    },
    passwordResetToken: {
      type: String,
      default: null,
    },
    passwordResetExpires: {
      type: Date,
      default: null,
    },
    // Legacy field — kept for backward compatibility with existing records.
    // New assignments are stored in ExamAssignment collection.
    assignedExamId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Exam",
      default: null,
    },
    isActive: {
      type: Boolean,
      default: true,
    },
    role: {
      type: String,
      default: "student",
      immutable: true,
    },
  },
  { timestamps: true }
);

module.exports = mongoose.model("Student", studentSchema);
