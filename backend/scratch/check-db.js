const mongoose = require("mongoose");
const path = require("path");
const dotenv = require("dotenv");

dotenv.config({ path: path.join(__dirname, "../.env") });

const dns = require("dns");
dns.setServers(["8.8.8.8", "8.8.4.4"]);

const ExamAssignment = require("../models/ExamAssignment");
const ExamSession = require("../models/ExamSession");
const Submission = require("../models/Submission");

async function check() {
  try {
    console.log("Connecting to Mongo...");
    await mongoose.connect(process.env.MONGO_URI);
    console.log("Connected to MongoDB successfully");

    const sessions = await ExamSession.find({});
    console.log(`Total exam sessions: ${sessions.length}`);
    
    const assignments = await ExamAssignment.find({});
    console.log(`Total assignments: ${assignments.length}`);

    for (const s of sessions) {
      console.log(`Session ID: ${s._id}, AssignmentID: ${s.assignmentId || 'None'}, Status: ${s.status}`);
      const sub = await Submission.findOne({ examSessionId: s._id });
      console.log(`  -> Submission Report: ${sub ? sub._id : 'None'}`);
    }
  } catch (err) {
    console.error("Check error:", err);
  } finally {
    await mongoose.disconnect();
  }
}

check();
