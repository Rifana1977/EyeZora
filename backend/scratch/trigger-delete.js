const mongoose = require("mongoose");
const path = require("path");
const dotenv = require("dotenv");

dotenv.config({ path: path.join(__dirname, "../.env") });

const dns = require("dns");
dns.setServers(["8.8.8.8", "8.8.4.4"]);

const controller = require("../controllers/assignmentController");

async function run() {
  try {
    await mongoose.connect(process.env.MONGO_URI);
    console.log("Connected to MongoDB");

    const req = {
      params: { id: "6a4e7172944c95a1f416af1d" }
    };

    const res = {
      status: function(code) {
        console.log(`Response Status: ${code}`);
        return this;
      },
      json: function(data) {
        console.log("Response JSON:", JSON.stringify(data, null, 2));
        return this;
      }
    };

    console.log("Triggering single delete...");
    await controller.deleteAssignment(req, res);

    console.log("\nTriggering bulk delete...");
    const reqBulk = {
      body: {
        assignmentIds: ["6a4e7172944c95a1f416af1d", "6a4e6e55e7080ef0ffd06379"]
      }
    };
    await controller.bulkDeleteAssignments(reqBulk, res);
  } catch (err) {
    console.error("Run error:", err);
  } finally {
    await mongoose.disconnect();
  }
}

run();
