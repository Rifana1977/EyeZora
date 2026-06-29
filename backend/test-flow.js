/**
 * test-flow.js
 * End-to-end integration test for the EyeZora proctoring pipeline.
 * Runs real HTTP requests against localhost:5000.
 */

const dns = require("dns");
dns.setServers(["8.8.8.8", "8.8.4.4"]);

const BASE_URL = "http://localhost:5000";

async function runTests() {
  console.log("🚀 Starting EyeZora End-to-End API Integration Tests...\n");

  let adminToken = "";
  let studentToken = "";
  let studentObjectId = "";
  let testStudentId = "STU_INTEG_99";
  let examId = "";
  let assignmentId = "";
  let sessionId = "";

  try {
    // ── 1. Admin Login ──────────────────────────────────────────────────────────
    console.log("⏳ Step 1: Logging in as Admin...");
    const adminLoginRes = await fetch(`${BASE_URL}/api/auth/admin/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: "admin@eyezora.com", password: "Admin@123" }),
    });

    if (!adminLoginRes.ok) {
      throw new Error(`Admin login failed: ${adminLoginRes.statusText}`);
    }

    const adminLoginData = await adminLoginRes.json();
    adminToken = adminLoginData.token;
    console.log("✅ Admin logged in successfully.\n");

    // ── 2. Create Student (Default Active) ──────────────────────────────────────
    console.log("⏳ Step 2: Registering a new student...");
    const regRes = await fetch(`${BASE_URL}/api/admin/students`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${adminToken}`,
      },
      body: JSON.stringify({
        studentId: testStudentId,
        name: "Integration Test Student",
        email: "test_integration@eyezora.com",
        password: "Student@123",
      }),
    });

    if (!regRes.ok) {
      const err = await regRes.json();
      throw new Error(`Student registration failed: ${err.error}`);
    }

    const studentData = await regRes.json();
    studentObjectId = studentData._id;
    console.log(`✅ Student created. ID: ${studentData.studentId}, Name: ${studentData.name}, isActive: ${studentData.isActive}`);
    if (studentData.isActive !== true) {
      throw new Error("FAIL: Newly created student should be Active by default.");
    }
    console.log("✅ Verified: Newly created student is Active by default.\n");

    // ── 3. Student Login (Should succeed) ────────────────────────────────────────
    console.log("⏳ Step 3: Verifying active student can log in...");
    const loginRes = await fetch(`${BASE_URL}/api/auth/student/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ identifier: testStudentId, password: "Student@123" }),
    });

    if (!loginRes.ok) {
      throw new Error(`Active student login failed: ${loginRes.statusText}`);
    }
    console.log("✅ Active student logged in successfully.\n");

    // ── 4. Deactivate Student ──────────────────────────────────────────────────
    console.log("⏳ Step 4: Deactivating student account...");
    const deactRes = await fetch(`${BASE_URL}/api/admin/students/${studentObjectId}`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${adminToken}`,
      },
      body: JSON.stringify({ isActive: false }),
    });

    if (!deactRes.ok) {
      throw new Error(`Failed to deactivate student: ${deactRes.statusText}`);
    }
    const deactData = await deactRes.json();
    console.log(`✅ Student updated. isActive: ${deactData.isActive}`);
    if (deactData.isActive !== false) {
      throw new Error("FAIL: Student isActive state should be false.");
    }

    // Try login again (Should fail with 403)
    console.log("⏳ Verifying inactive student login is blocked...");
    const failLoginRes = await fetch(`${BASE_URL}/api/auth/student/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ identifier: testStudentId, password: "Student@123" }),
    });

    console.log(`ℹ️ Inactive Login Status Code: ${failLoginRes.status}`);
    if (failLoginRes.status === 403) {
      const failLoginData = await failLoginRes.json();
      console.log(`✅ Inactive login successfully blocked. Error message: "${failLoginData.error}"`);
    } else {
      throw new Error(`FAIL: Inactive student login should return 403 Forbidden. Got ${failLoginRes.status}`);
    }
    console.log("✅ Verified: Inactive student login is correctly blocked.\n");

    // ── 5. Reactivate Student ──────────────────────────────────────────────────
    console.log("⏳ Step 5: Reactivating student account...");
    const reactRes = await fetch(`${BASE_URL}/api/admin/students/${studentObjectId}`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${adminToken}`,
      },
      body: JSON.stringify({ isActive: true }),
    });

    if (!reactRes.ok) {
      throw new Error(`Failed to reactivate student: ${reactRes.statusText}`);
    }
    const reactData = await reactRes.json();
    console.log(`✅ Student updated. isActive: ${reactData.isActive}`);
    if (reactData.isActive !== true) {
      throw new Error("FAIL: Student isActive state should be true.");
    }
    console.log("✅ Verified: Student reactivated successfully.\n");

    // ── 6. Create Exam ───────────────────────────────────────────────────────────
    console.log("⏳ Step 6: Creating a test exam...");
    const examRes = await fetch(`${BASE_URL}/api/admin/exam`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${adminToken}`,
      },
      body: JSON.stringify({ title: "Integration Test Chemistry", duration: 45 }),
    });

    if (!examRes.ok) {
      throw new Error(`Failed to create exam: ${examRes.statusText}`);
    }
    const examData = await examRes.json();
    examId = examData._id;
    console.log(`✅ Exam created. ID: ${examId}, Title: "${examData.title}", Duration: ${examData.duration} minutes.\n`);

    // ── 7. Assign Exam (Auto-Calculate End Time) ──────────────────────────────────
    console.log("⏳ Step 7: Assigning exam to student (testing auto-calculated end time)...");
    const startTimeStr = "2026-06-29T10:00:00.000Z";
    const assignRes = await fetch(`${BASE_URL}/api/admin/assignments`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${adminToken}`,
      },
      body: JSON.stringify({
        studentObjectId,
        examId,
        startTime: startTimeStr,
        duration: 45, // 45 minutes
        notes: "Automated test assignment",
      }),
    });

    if (!assignRes.ok) {
      const err = await assignRes.json();
      throw new Error(`Failed to assign exam: ${err.error}`);
    }

    const assignData = await assignRes.json();
    assignmentId = assignData._id;
    console.log(`✅ Exam assigned. StartTime: ${assignData.startTime}`);
    console.log(`✅ Calculated EndTime in response: ${assignData.endTime}`);

    // Verify mathematical calculation
    const expectedEndTime = new Date(new Date(startTimeStr).getTime() + 45 * 60000).toISOString();
    if (new Date(assignData.endTime).toISOString() === expectedEndTime) {
      console.log("✅ Verified: End Time matches precisely (StartTime + 45 mins).");
    } else {
      throw new Error(`FAIL: Calculated EndTime mismatch. Expected: ${expectedEndTime}, Got: ${assignData.endTime}`);
    }
    console.log("✅ Verified: Exam scheduling auto-calculation works perfectly.\n");

    // ── 8. Student Login (Get Assignment) ───────────────────────────────────────
    console.log("⏳ Step 8: Logging in student to fetch assignment details...");
    const stdLoginRes = await fetch(`${BASE_URL}/api/auth/student/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ identifier: testStudentId, password: "Student@123" }),
    });

    if (!stdLoginRes.ok) {
      throw new Error(`Student login failed: ${stdLoginRes.statusText}`);
    }
    const stdLoginData = await stdLoginRes.json();
    studentToken = stdLoginData.token;
    console.log("✅ Student logged in.");
    console.log(`✅ Assigned Exam in response: "${stdLoginData.user.assignedExam?.title}" (Assignment ID: ${stdLoginData.user.assignedExam?.assignmentId})\n`);

    // ── 9. Start Proctoring Session ──────────────────────────────────────────────
    console.log("⏳ Step 9: Starting the student exam proctoring session...");
    const sessStartRes = await fetch(`${BASE_URL}/api/session/start`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${studentToken}`,
      },
      body: JSON.stringify({
        examId,
        examTitle: "Integration Test Chemistry",
        assignmentId,
      }),
    });

    if (!sessStartRes.ok) {
      throw new Error(`Failed to start session: ${sessStartRes.statusText}`);
    }
    const sessStartData = await sessStartRes.json();
    sessionId = sessStartData.sessionId;
    console.log(`✅ Session started. SessionID: ${sessionId}\n`);

    // ── 10. Log Proctoring Violations ────────────────────────────────────────────
    console.log("⏳ Step 10: Logging proctoring violations...");
    const violationsToLog = [
      { event: "FULLSCREEN_EXIT", confidence: 100 },
      { event: "TAB_SWITCH", confidence: 100 },
      { event: "CAMERA_DISCONNECTED", confidence: 100 },
      { event: "MICROPHONE_DISABLED", confidence: 100 },
      { event: "NO_FACE", confidence: 95 },
    ];

    for (const v of violationsToLog) {
      const logRes = await fetch(`${BASE_URL}/api/session/log`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${studentToken}`,
        },
        body: JSON.stringify({
          sessionId,
          event: v.event,
          confidence: v.confidence,
        }),
      });

      if (!logRes.ok) {
        throw new Error(`Failed to log violation ${v.event}: ${logRes.statusText}`);
      }
      const logData = await logRes.json();
      console.log(`✅ Logged event: "${v.event}", Severity: ${logData.severity}, Session Risk: ${logData.riskLevel}`);
    }
    console.log("✅ Verified: Real-time proctoring log events submitted successfully.\n");

    // ── 11. End Session (Submit Exam) ────────────────────────────────────────────
    console.log("⏳ Step 11: Submitting student exam...");
    const sessEndRes = await fetch(`${BASE_URL}/api/session/end`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${studentToken}`,
      },
      body: JSON.stringify({
        sessionId,
        answers: [-1, -1, -1], // placeholder answers
        examId,
      }),
    });

    if (!sessEndRes.ok) {
      throw new Error(`Failed to submit exam: ${sessEndRes.statusText}`);
    }
    const sessEndData = await sessEndRes.json();
    console.log(`✅ Exam submitted. Status: ${sessEndData.status}, Score: ${sessEndData.score}/${sessEndData.totalMarks}\n`);

    // ── 12. Admin Verifies Session Counters ──────────────────────────────────────
    console.log("⏳ Step 12: Admin retrieving exam session report to verify counters...");
    const reportRes = await fetch(`${BASE_URL}/api/admin/sessions/${sessionId}/report`, {
      headers: {
        "Authorization": `Bearer ${adminToken}`,
      },
    });

    if (!reportRes.ok) {
      throw new Error(`Failed to get report: ${reportRes.statusText}`);
    }

    const reportData = await reportRes.json();
    const sess = reportData.session;
    const logs = reportData.logs;

    console.log(`✅ Session Report retrieved. Risk Level: ${sess.riskLevel}`);
    console.log(`✅ Total violations registered in DB session: ${sess.totalViolations}`);
    console.log(`✅ Fullscreen Exit count: ${sess.fullscreenExitCount}`);
    console.log(`✅ Tab Switch count: ${sess.tabSwitchCount}`);
    console.log(`✅ No Face count: ${sess.noFaceCount}`);
    console.log(`✅ Total logs recorded in proctoring logs table: ${logs.length}`);

    // Assert counts
    if (sess.fullscreenExitCount !== 1 || sess.tabSwitchCount !== 1 || sess.noFaceCount !== 1) {
      throw new Error("FAIL: Bounding event counters in session do not match logged values.");
    }
    if (logs.length < 5) {
      throw new Error(`FAIL: Expected at least 5 logs, got ${logs.length}`);
    }

    console.log("\n⭐️ ALL E2E API VERIFICATION TESTS PASSED SUCCESSFULLY! ⭐️\n");

  } catch (err) {
    console.error("\n❌ TEST RUN FAILURE:", err.message);
  } finally {
    // ── Cleanup Test Records from Database ────────────────────────────────────
    console.log("⏳ Cleanup: Removing test student and assignment records...");
    try {
      if (studentObjectId) {
        await fetch(`${BASE_URL}/api/admin/students/${studentObjectId}`, {
          method: "DELETE",
          headers: { "Authorization": `Bearer ${adminToken}` },
        });
      }
      if (examId) {
        await fetch(`${BASE_URL}/api/admin/exam/${examId}`, {
          method: "DELETE",
          headers: { "Authorization": `Bearer ${adminToken}` },
        });
      }
      console.log("🧹 Cleanup complete. Environment is clean.");
    } catch (cleanupErr) {
      console.error("⚠️ Cleanup warning:", cleanupErr.message);
    }
  }
}

runTests();
