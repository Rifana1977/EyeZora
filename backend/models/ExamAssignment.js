const mongoose = require("mongoose");

/**
 * ExamAssignment — Dedicated collection that links a student to an exam.
 * One document per assignment event. Historical records are never deleted.
 * Multiple assignments per student are supported (e.g., retakes, rescheduling).
 */
const examAssignmentSchema = new mongoose.Schema(
  {
    studentId: {
      type: String,
      required: true,
      index: true,
      // References Student.studentId (string key, not ObjectId)
    },
    studentObjectId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Student",
      required: true,
    },
    examId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Exam",
      required: true,
    },
    assignedBy: {
      type: String,
      default: "admin",
    },
    assignedAt: {
      type: Date,
      default: Date.now,
    },
    startTime: {
      type: Date,
      required: true,
    },
    endTime: {
      type: Date,
      required: true,
    },
    duration: {
      type: Number, // minutes — overrides exam default if set
      required: true,
    },
    status: {
      type: String,
      enum: [
        "assigned",   // Created but not yet time to start
        "upcoming",   // Within 24h of start
        "started",    // Student has clicked Start Exam
        "completed",  // Exam session ended successfully
        "expired",    // endTime passed without completion
        "cancelled",  // Manually cancelled by admin
      ],
      default: "assigned",
    },
    notes: {
      type: String,
      default: "",
    },
  },
  { timestamps: true }
);

// Compound index: find active assignment for a student quickly
examAssignmentSchema.index({ studentId: 1, status: 1 });
examAssignmentSchema.index({ examId: 1, status: 1 });

module.exports = mongoose.model("ExamAssignment", examAssignmentSchema);
