"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { getUser, clearAuth } from "@/lib/auth";
import { studentApi, sessionApi, aiApi } from "@/lib/api";
import { toast } from "@/app/components/ui/Toast";

interface Question {
  _id: string;
  questionNumber: number;
  questionText: string;
  options: string[];
  marks: number;
}

type ExamPhase = "loading" | "exam" | "submitting" | "done";

const OPTION_LABELS = ["A", "B", "C", "D", "E"];

export default function StudentExamPage() {
  const router = useRouter();
  const user = getUser();

  // ── Media Refs ─────────────────────────────────────────────────────────────
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const videoRecorderRef = useRef<MediaRecorder | null>(null);
  const audioRecorderRef = useRef<MediaRecorder | null>(null);
  const videoChunksRef = useRef<Blob[]>([]);
  const audioChunksRef = useRef<Blob[]>([]);

  // ── Session Refs ────────────────────────────────────────────────────────────
  const sessionIdRef = useRef<string>("");
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const aiIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isSubmittingRef = useRef<boolean>(false);
  const focusLostRef = useRef<boolean>(false); // Track focus state to avoid duplicate RESTORED logs

  // ── State ──────────────────────────────────────────────────────────────────
  const [phase, setPhase] = useState<ExamPhase>("loading");
  const [questions, setQuestions] = useState<Question[]>([]);
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [currentQ, setCurrentQ] = useState(0);
  const [timeLeft, setTimeLeft] = useState(0);
  const [violations, setViolations] = useState<string[]>([]);
  const [warningMsg, setWarningMsg] = useState<string | null>(null);
  const [tabSwitches, setTabSwitches] = useState(0);
  const [fsExits, setFsExits] = useState(0);
  const [isFsActive, setIsFsActive] = useState(true);

  const examId = user?.assignedExam?.id ?? "";
  const examTitle = user?.assignedExam?.title ?? "Exam";
  const examDuration = user?.assignedExam?.duration ?? 60;
  const assignmentId = user?.assignedExam?.assignmentId ?? undefined;

  // ── Cleanup all resources ──────────────────────────────────────────────────
  const cleanupResources = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRecorderRef.current && videoRecorderRef.current.state !== "inactive") {
      videoRecorderRef.current.stop();
    }
    if (audioRecorderRef.current && audioRecorderRef.current.state !== "inactive") {
      audioRecorderRef.current.stop();
    }
    if (aiIntervalRef.current) {
      clearInterval(aiIntervalRef.current);
      aiIntervalRef.current = null;
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    }
  }, []);

  // ── Log violation ──────────────────────────────────────────────────────────
  const logViolation = useCallback(async (event: string, confidence = 100) => {
    if (!sessionIdRef.current) return;
    try {
      await sessionApi.logEvent(sessionIdRef.current, event, confidence);
    } catch { /* silent fail — non-blocking */ }

    setViolations((v) => [...v.slice(-10), event]);
    showWarning(event);
  }, []);

  function showWarning(event: string) {
    const msgs: Record<string, string> = {
      MULTIPLE_FACES: "🚨 Multiple faces detected!",
      NO_FACE: "⚠️ Face not visible — please stay in frame",
      LOOKING_AWAY: "👀 Please look at the screen",
      PHONE_DETECTED: "📱 Phone detected — prohibited!",
      TAB_SWITCH: "🔀 Browser tab switch detected!",
      WINDOW_FOCUS_LOST: "⚠️ Window lost focus — please stay in exam",
      FULLSCREEN_EXIT: "⛶ Full-screen mode is required!",
    };
    const msg = msgs[event];
    if (msg) {
      setWarningMsg(msg);
      setTimeout(() => setWarningMsg(null), 3500);
    }
  }

  // ── Fullscreen Management ──────────────────────────────────────────────────
  const enterFullscreen = useCallback(() => {
    const el = document.documentElement;
    if (el.requestFullscreen) {
      el.requestFullscreen({ navigationUI: "hide" })
        .then(() => setIsFsActive(true))
        .catch(() => setIsFsActive(false));
    }
  }, []);

  const restoreFullscreen = useCallback(async () => {
    try {
      await document.documentElement.requestFullscreen({ navigationUI: "hide" });
      setIsFsActive(true);
    } catch {
      toast.error("Fullscreen restoration failed. Please try again.");
    }
  }, []);

  useEffect(() => {
    const onFsChange = () => {
      const active = !!document.fullscreenElement;
      setIsFsActive(active);
      if (!active && phase === "exam") {
        setFsExits((n) => n + 1);
        logViolation("FULLSCREEN_EXIT", 100);
      }
    };
    document.addEventListener("fullscreenchange", onFsChange);
    return () => document.removeEventListener("fullscreenchange", onFsChange);
  }, [phase, logViolation]);

  // ── Tab / Window Focus Monitoring ──────────────────────────────────────────
  useEffect(() => {
    if (phase !== "exam") return;

    const onVisibilityChange = () => {
      if (document.hidden) {
        setTabSwitches((n) => n + 1);
        logViolation("TAB_SWITCH", 100);
      }
    };

    const onBlur = () => {
      // Only log if not already in a focus-lost state
      if (!focusLostRef.current && !document.hidden) {
        focusLostRef.current = true;
        logViolation("WINDOW_FOCUS_LOST", 100);
      }
    };

    const onFocus = () => {
      if (focusLostRef.current) {
        focusLostRef.current = false;
        logViolation("WINDOW_FOCUS_RESTORED", 100);
      }
    };

    document.addEventListener("visibilitychange", onVisibilityChange);
    window.addEventListener("blur", onBlur);
    window.addEventListener("focus", onFocus);

    return () => {
      document.removeEventListener("visibilitychange", onVisibilityChange);
      window.removeEventListener("blur", onBlur);
      window.removeEventListener("focus", onFocus);
    };
  }, [phase, logViolation]);

  // ── Block Back Button ──────────────────────────────────────────────────────
  useEffect(() => {
    if (phase !== "exam") return;
    history.pushState(null, "", location.href);
    window.onpopstate = () => history.go(1);
    return () => { window.onpopstate = null; };
  }, [phase]);

  // ── Initialize from pre-exam permissions ──────────────────────────────────
  useEffect(() => {
    setIsFsActive(!!document.fullscreenElement);
    if (!user || !examId) {
      toast.error("No exam assigned. Please login again.");
      router.push("/student/login");
      return;
    }

    // Check if coming from pre-exam page with permissions already granted
    const examReady = typeof window !== "undefined" && sessionStorage.getItem("ez_exam_ready") === "true";
    const existingStream = typeof window !== "undefined" ? (window as any).__eyezoraStream as MediaStream | undefined : undefined;

    if (examReady && existingStream) {
      sessionStorage.removeItem("ez_exam_ready");
      delete (window as any).__eyezoraStream;
      streamRef.current = existingStream;
      if (videoRef.current) {
        videoRef.current.srcObject = existingStream;
      }
      setupRecorders(existingStream);
      startExamSession();
    } else {
      // Fallback: start permissions from scratch
      initExam();
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Global Cleanup on Unmount ──────────────────────────────────────────────
  useEffect(() => {
    return () => { cleanupResources(); };
  }, [cleanupResources]);

  // ── Setup Media Recorders ──────────────────────────────────────────────────
  function setupRecorders(stream: MediaStream) {
    // Video recorder (captures all tracks)
    const videoTracks = stream.getVideoTracks();
    const audioTracks = stream.getAudioTracks();

    // Attach listeners to detect camera and mic disconnects/mutes
    videoTracks.forEach((track) => {
      track.onmute = () => {
        logViolation("CAMERA_DISCONNECTED", 100);
      };
      track.onended = () => {
        logViolation("CAMERA_DISCONNECTED", 100);
      };
    });

    audioTracks.forEach((track) => {
      track.onmute = () => {
        logViolation("MICROPHONE_DISABLED", 100);
      };
      track.onended = () => {
        logViolation("MICROPHONE_DISABLED", 100);
      };
    });

    if (videoTracks.length > 0) {
      const videoOnlyStream = new MediaStream([...videoTracks]);
      const videoRecorder = new MediaRecorder(videoOnlyStream, { mimeType: "video/webm" });
      videoRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) videoChunksRef.current.push(e.data);
      };
      videoRecorder.start(5000);
      videoRecorderRef.current = videoRecorder;
    }

    // Separate audio recorder
    if (audioTracks.length > 0) {
      const audioOnlyStream = new MediaStream([...audioTracks]);
      try {
        const audioRecorder = new MediaRecorder(audioOnlyStream, { mimeType: "audio/webm" });
        audioRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) audioChunksRef.current.push(e.data);
        };
        audioRecorder.start(5000);
        audioRecorderRef.current = audioRecorder;
      } catch {
        // Fallback without specific mimeType
        const audioRecorder = new MediaRecorder(audioOnlyStream);
        audioRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) audioChunksRef.current.push(e.data);
        };
        audioRecorder.start(5000);
        audioRecorderRef.current = audioRecorder;
      }
    }
  }

  // ── Init Exam (fallback if not coming from pre-exam) ──────────────────────
  async function initExam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setupRecorders(stream);

      // Request fullscreen
      try {
        await document.documentElement.requestFullscreen();
      } catch {
        toast.error("Fullscreen is required. Please allow fullscreen access.");
        return;
      }

      await startExamSession();
    } catch (err) {
      toast.error("Camera and microphone permissions are required.");
      router.push("/student/pre-exam");
    }
  }

  // ── Start Exam Session ─────────────────────────────────────────────────────
  async function startExamSession() {
    try {
      const qs = await studentApi.getExamQuestions(examId);
      setQuestions(qs);
      setTimeLeft(examDuration * 60);
    } catch (e: any) {
      toast.error("Failed to load exam questions: " + e.message);
      return;
    }

    try {
      const { sessionId } = await sessionApi.start(examId, examTitle, assignmentId);
      sessionIdRef.current = sessionId;
    } catch (e: any) {
      toast.error("Failed to start exam session: " + e.message);
      return;
    }

    // Ensure fullscreen
    if (!document.fullscreenElement) {
      enterFullscreen();
      setIsFsActive(false);
      logViolation("FULLSCREEN_EXIT", 100);
    }

    setPhase("exam");
    startTimer();
    startAILoop();
  }

  // ── Timer ──────────────────────────────────────────────────────────────────
  function startTimer() {
    timerRef.current = setInterval(() => {
      setTimeLeft((t) => {
        if (t <= 1) {
          clearInterval(timerRef.current!);
          submitExam();
          return 0;
        }
        return t - 1;
      });
    }, 1000);
  }

  // ── AI Detection Loop ──────────────────────────────────────────────────────
  function startAILoop() {
    if (aiIntervalRef.current) clearInterval(aiIntervalRef.current);

    aiIntervalRef.current = setInterval(async () => {
      const video = videoRef.current;

      // Proactive check if camera stream is disconnected/disabled
      if (streamRef.current) {
        const videoTrack = streamRef.current.getVideoTracks()[0];
        if (!videoTrack || videoTrack.readyState === "ended" || !videoTrack.enabled) {
          logViolation("CAMERA_DISCONNECTED", 100);
        }
      }

      if (!video || video.readyState < 2) return;

      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth || 320;
      canvas.height = video.videoHeight || 240;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(video, 0, 0);
      const base64 = canvas.toDataURL("image/jpeg", 0.7);

      const result = await aiApi.analyzeFrame(
        base64,
        user?.studentId ?? "",
        examId,
        sessionIdRef.current
      );

      if (result?.events) {
        for (const ev of result.events) {
          await logViolation(ev.event, ev.confidence);
        }
      }
    }, 1000);
  }

  // ── Submit Exam ────────────────────────────────────────────────────────────
  async function submitExam() {
    if (isSubmittingRef.current) return;
    isSubmittingRef.current = true;
    setPhase("submitting");

    // Clear intervals
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    if (aiIntervalRef.current) { clearInterval(aiIntervalRef.current); aiIntervalRef.current = null; }

    // Stop media stream (releases camera hardware)
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    // Exit fullscreen
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    }

    // Collect final video blob
    const stopVideoPromise = new Promise<Blob | null>((resolve) => {
      const recorder = videoRecorderRef.current;
      if (recorder && recorder.state !== "inactive") {
        recorder.onstop = () => {
          resolve(videoChunksRef.current.length > 0
            ? new Blob(videoChunksRef.current, { type: "video/webm" })
            : null);
        };
        recorder.stop();
      } else {
        resolve(videoChunksRef.current.length > 0
          ? new Blob(videoChunksRef.current, { type: "video/webm" })
          : null);
      }
    });

    // Collect final audio blob
    const stopAudioPromise = new Promise<Blob | null>((resolve) => {
      const recorder = audioRecorderRef.current;
      if (recorder && recorder.state !== "inactive") {
        recorder.onstop = () => {
          resolve(audioChunksRef.current.length > 0
            ? new Blob(audioChunksRef.current, { type: "audio/webm" })
            : null);
        };
        recorder.stop();
      } else {
        resolve(audioChunksRef.current.length > 0
          ? new Blob(audioChunksRef.current, { type: "audio/webm" })
          : null);
      }
    });

    try {
      const answersArray: number[] = questions.map((q) =>
        answers[q._id] !== undefined ? answers[q._id] : -1
      );

      const [result, videoBlob, audioBlob] = await Promise.all([
        sessionApi.end(sessionIdRef.current, answersArray, examId),
        stopVideoPromise,
        stopAudioPromise,
      ]);

      // Upload recordings in background (non-blocking)
      if (videoBlob || audioBlob) {
        sessionApi.uploadRecording(sessionIdRef.current, videoBlob, audioBlob)
          .then(() => console.log("Recordings uploaded successfully"))
          .catch((err) => console.error("Recording upload failed:", err));
      }

      toast.success(`Exam submitted! Score: ${result.score}/${result.totalMarks}`);
      setPhase("done");
    } catch (err: any) {
      toast.error("Submission failed: " + err.message);
      isSubmittingRef.current = false;
      setPhase("exam");
    }
  }

  // ── Exit and Logout ────────────────────────────────────────────────────────
  const handleExit = () => {
    cleanupResources();
    clearAuth();
    document.cookie = "ez_token=; Max-Age=0; path=/";
    document.cookie = "ez_role=; Max-Age=0; path=/";
    router.replace("/student/login");
  };

  function formatTime(sec: number) {
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    return h > 0
      ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
      : `${m}:${String(s).padStart(2, "0")}`;
  }

  const timerColor = timeLeft < 300 ? "#ef4444" : timeLeft < 600 ? "#f59e0b" : "#10b981";

  // ── PHASES ─────────────────────────────────────────────────────────────────

  if (phase === "loading") {
    return (
      <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "var(--bg-base)" }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ width: 48, height: 48, border: "3px solid rgba(124,58,237,0.3)", borderTopColor: "#7c3aed", borderRadius: "50%", animation: "spin 0.8s linear infinite", margin: "0 auto 16px" }} />
          <p style={{ color: "var(--text-muted)" }}>Initializing exam environment…</p>
        </div>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  if (phase === "done") {
    return (
      <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "var(--bg-base)", padding: 20 }}>
        <div className="glass-card fade-in" style={{ padding: 48, textAlign: "center", maxWidth: 480 }}>
          <div style={{ fontSize: 60, marginBottom: 20 }}>✅</div>
          <h2 style={{ color: "var(--text-primary)", fontSize: 26, fontWeight: 800, marginBottom: 8 }}>
            Exam Submitted!
          </h2>
          <p style={{ color: "var(--text-muted)", fontSize: 14, marginBottom: 28, lineHeight: 1.7 }}>
            Your answers have been recorded and your session has ended. Your invigilator will review the proctoring report.
          </p>
          <button
            onClick={handleExit}
            className="btn-glow"
            style={{ padding: "12px 28px", borderRadius: 12, fontSize: 15, fontWeight: 700 }}
          >
            Exit
          </button>
        </div>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  if (phase === "submitting") {
    return (
      <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "var(--bg-base)" }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ width: 48, height: 48, border: "3px solid rgba(124,58,237,0.3)", borderTopColor: "#7c3aed", borderRadius: "50%", animation: "spin 0.8s linear infinite", margin: "0 auto 16px" }} />
          <p style={{ color: "var(--text-primary)", fontSize: 16, fontWeight: 700 }}>Submitting your exam…</p>
          <p style={{ color: "var(--text-muted)", fontSize: 13, marginTop: 8 }}>Please wait while we save your recording.</p>
        </div>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  // ── EXAM PHASE ─────────────────────────────────────────────────────────────
  const q = questions[currentQ];
  const answered = Object.keys(answers).length;

  return (
    <div style={{
      minHeight: "100vh",
      background: "var(--bg-base)",
      display: "flex",
      flexDirection: "column",
      overflow: "hidden",
    }}>
      {/* Hidden video element for AI analysis — active but invisible */}
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          width: 1,
          height: 1,
          opacity: 0.001,
          pointerEvents: "none",
        }}
        aria-hidden="true"
      />

      {/* Fullscreen Overlay */}
      {!isFsActive && (
        <div style={{
          position: "fixed",
          top: 0, left: 0, right: 0, bottom: 0,
          background: "var(--bg-base)",
          color: "var(--text-primary)",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 99999,
          padding: 24,
          textAlign: "center",
        }}>
          <div style={{ fontSize: 64, marginBottom: 20 }}>⚠️</div>
          <h2 style={{ fontSize: 28, fontWeight: 800, marginBottom: 12, color: "var(--danger)" }}>
            Fullscreen Exited
          </h2>
          <p style={{ color: "var(--text-secondary)", fontSize: 16, maxWidth: 480, lineHeight: 1.6, marginBottom: 32 }}>
            Exiting fullscreen mode is a proctoring violation and has been logged.
            You must return to fullscreen mode to continue your examination.
          </p>
          <button
            onClick={restoreFullscreen}
            className="btn-glow"
            style={{
              padding: "16px 36px",
              borderRadius: 14,
              fontSize: 16,
              fontWeight: 700,
            }}
          >
            Restore Fullscreen ⛶
          </button>
        </div>
      )}

      {/* Warning Banner */}
      {warningMsg && (
        <div style={{
          position: "fixed",
          top: 0, left: 0, right: 0,
          background: "linear-gradient(90deg,#b91c1c,#ef4444)",
          color: "#fff",
          textAlign: "center",
          padding: "12px 0",
          fontSize: 15,
          fontWeight: 700,
          zIndex: 9999,
          animation: "slideInDown 0.3s ease",
        }}>
          {warningMsg}
        </div>
      )}

      {/* Top Bar */}
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 24px",
        height: 60,
        background: "var(--bg-surface)",
        borderBottom: "1px solid var(--bg-border)",
        flexShrink: 0,
      }}>
        {/* Brand */}
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 20 }}>👁</span>
          <span style={{ color: "var(--text-accent)", fontWeight: 800, fontSize: 16, letterSpacing: 1 }}>EYEZORA</span>
          <span style={{ color: "var(--text-muted)", fontSize: 12, marginLeft: 8 }}>{examTitle}</span>
        </div>

        {/* Timer */}
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          background: "var(--input-bg)",
          border: `1px solid ${timerColor}40`,
          borderRadius: 10,
          padding: "7px 16px",
        }}>
          <span style={{ fontSize: 16 }}>⏱</span>
          <span style={{ color: timerColor, fontSize: 18, fontWeight: 800, fontFamily: "monospace" }}>
            {formatTime(timeLeft)}
          </span>
        </div>

        {/* Stats */}
        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          {tabSwitches > 0 && (
            <span style={{ color: "#ef4444", fontSize: 12, fontWeight: 700 }}>
              🔀 {tabSwitches} switches
            </span>
          )}
          {fsExits > 0 && (
            <span style={{ color: "#ef4444", fontSize: 12, fontWeight: 700 }}>
              ⛶ {fsExits} exits
            </span>
          )}
          <span style={{ color: "var(--text-muted)", fontSize: 12 }}>
            {answered}/{questions.length} answered
          </span>
          {/* AI monitoring active indicator */}
          <span style={{ display: "flex", alignItems: "center", gap: 4, color: "#10b981", fontSize: 11, fontWeight: 700 }}>
            <span className="pulse-dot" style={{ fontSize: 8 }}>●</span>
            AI ACTIVE
          </span>
        </div>
      </div>

      {/* Main Area */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        {/* Question Panel */}
        <div style={{ flex: 1, padding: "28px 32px", overflowY: "auto" }}>
          {/* Progress */}
          <div style={{ marginBottom: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
              <span style={{ color: "var(--text-muted)", fontSize: 12 }}>Question {currentQ + 1} of {questions.length}</span>
              <span style={{ color: "var(--text-muted)", fontSize: 12 }}>
                {q?.marks ?? 1} mark{(q?.marks ?? 1) !== 1 ? "s" : ""}
              </span>
            </div>
            <div style={{ height: 4, background: "var(--bg-hover)", borderRadius: 2 }}>
              <div style={{
                height: "100%",
                width: `${((currentQ + 1) / questions.length) * 100}%`,
                background: "linear-gradient(90deg,#7c3aed,#a78bfa)",
                borderRadius: 2,
                transition: "width 0.3s",
              }} />
            </div>
          </div>

          {/* Question */}
          {q && (
            <div className="glass-card" style={{ padding: 28, marginBottom: 20, background: "var(--card-bg)" }}>
              <p style={{ color: "var(--text-primary)", fontSize: 17, fontWeight: 600, lineHeight: 1.6, marginBottom: 24 }}>
                {currentQ + 1}. {q.questionText}
              </p>

              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {q.options.map((opt, idx) => {
                  const selected = answers[q._id] === idx;
                  return (
                    <button
                      key={idx}
                      onClick={() => setAnswers((a) => ({ ...a, [q._id]: idx }))}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 14,
                        padding: "14px 18px",
                        borderRadius: 12,
                        border: selected
                          ? "2px solid rgba(124,58,237,0.7)"
                          : "1px solid var(--bg-border)",
                        background: selected
                          ? "var(--table-header-bg)"
                          : "var(--bg-elevated)",
                        cursor: "pointer",
                        textAlign: "left",
                        transition: "all 0.15s",
                      }}
                      onMouseOver={(e) => { if (!selected) e.currentTarget.style.background = "var(--bg-hover)"; }}
                      onMouseOut={(e) => { if (!selected) e.currentTarget.style.background = "var(--bg-elevated)"; }}
                    >
                      <span style={{
                        width: 32, height: 32, borderRadius: 8, flexShrink: 0,
                        background: selected ? "rgba(124,58,237,0.4)" : "var(--bg-hover)",
                        color: selected ? "var(--text-accent)" : "var(--text-muted)",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 13, fontWeight: 800,
                      }}>
                        {OPTION_LABELS[idx]}
                      </span>
                      <span style={{ color: selected ? "var(--text-accent)" : "var(--text-secondary)", fontSize: 14, fontWeight: selected ? 600 : 400 }}>
                        {opt}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Navigation */}
          <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
            <button
              onClick={() => setCurrentQ((q) => Math.max(0, q - 1))}
              disabled={currentQ === 0}
              style={{
                padding: "11px 22px", borderRadius: 10,
                background: "var(--bg-elevated)", border: "1px solid var(--bg-border)",
                color: currentQ === 0 ? "var(--text-muted)" : "var(--text-primary)",
                cursor: currentQ === 0 ? "not-allowed" : "pointer",
                fontWeight: 600, fontSize: 14,
              }}
            >
              ← Previous
            </button>

            {/* Question dot nav */}
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap", justifyContent: "center", flex: 1 }}>
              {questions.map((_, i) => (
                <button
                  key={i}
                  onClick={() => setCurrentQ(i)}
                  style={{
                    width: 32, height: 32, borderRadius: 8, border: "none",
                    background: i === currentQ
                      ? "linear-gradient(135deg,#7c3aed,#5b21b6)"
                      : answers[questions[i]._id] !== undefined
                      ? "rgba(16,185,129,0.25)"
                      : "var(--input-bg)",
                    color: i === currentQ ? "var(--text-primary)" : answers[questions[i]._id] !== undefined ? "var(--success)" : "var(--text-muted)",
                    cursor: "pointer", fontSize: 12, fontWeight: 700,
                  }}
                >
                  {i + 1}
                </button>
              ))}
            </div>

            {currentQ < questions.length - 1 ? (
              <button
                onClick={() => setCurrentQ((q) => Math.min(questions.length - 1, q + 1))}
                className="btn-glow"
                style={{ padding: "11px 22px", borderRadius: 10, fontSize: 14 }}
              >
                Next →
              </button>
            ) : (
              <button
                onClick={submitExam}
                id="submit-exam-btn"
                style={{
                  padding: "11px 22px", borderRadius: 10, fontSize: 14, fontWeight: 700,
                  background: "linear-gradient(135deg,#059669,#10b981)",
                  color: "#fff", border: "none", cursor: "pointer",
                  transition: "all 0.15s",
                }}
              >
                ✓ Submit Exam
              </button>
            )}
          </div>
        </div>

        {/* Right Panel — AI Status (NO camera preview) */}
        <div style={{
          width: 220,
          background: "var(--bg-surface)",
          borderLeft: "1px solid var(--bg-border)",
          padding: 16,
          display: "flex",
          flexDirection: "column",
          gap: 16,
          flexShrink: 0,
        }}>
          {/* AI Monitor Status */}
          <div>
            <p style={{ color: "var(--text-accent)", fontSize: 11, fontWeight: 700, letterSpacing: "0.07em", textTransform: "uppercase", marginBottom: 10 }}>
              AI Monitor
            </p>
            <div style={{
              padding: "14px",
              borderRadius: 12,
              background: "rgba(16,185,129,0.06)",
              border: "1px solid rgba(16,185,129,0.2)",
              marginBottom: 10,
              textAlign: "center",
            }}>
              <div style={{ fontSize: 24, marginBottom: 4 }}>🎥</div>
              <div style={{ color: "var(--success)", fontSize: 12, fontWeight: 700 }}>Camera Active</div>
              <div style={{ color: "var(--text-muted)", fontSize: 10, marginTop: 2 }}>Monitoring in progress</div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {[
                { label: "Face Detection",   active: true },
                { label: "Object Detection", active: true },
                { label: "Gaze Analysis",    active: true },
                { label: "Mic Recording",    active: true },
                { label: "Video Recording",  active: true },
              ].map((item) => (
                <div key={item.label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ color: "var(--text-muted)", fontSize: 11 }}>{item.label}</span>
                  <span style={{ color: "var(--success)", fontSize: 10, fontWeight: 700, display: "flex", alignItems: "center", gap: 3 }}>
                    <span className="pulse-dot" style={{ fontSize: 7 }}>●</span>
                    ON
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Recent Alerts */}
          {violations.length > 0 && (
            <div>
              <p style={{ color: "#ef4444", fontSize: 11, fontWeight: 700, letterSpacing: "0.07em", textTransform: "uppercase", marginBottom: 8 }}>
                Recent Alerts
              </p>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                {violations.slice(-5).reverse().map((v, i) => (
                  <div key={i} style={{
                    padding: "5px 8px",
                    borderRadius: 6,
                    background: "rgba(239,68,68,0.08)",
                    border: "1px solid rgba(239,68,68,0.2)",
                    color: "#ef4444",
                    fontSize: 10,
                    fontWeight: 600,
                  }}>
                    {v.replace(/_/g, " ")}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Final Submit at bottom */}
          <div style={{ marginTop: "auto" }}>
            <button
              onClick={submitExam}
              id="final-submit-btn"
              style={{
                width: "100%",
                padding: "11px 0",
                borderRadius: 10,
                background: "rgba(16,185,129,0.10)",
                border: "1px solid rgba(16,185,129,0.25)",
                color: "#10b981",
                fontWeight: 700,
                fontSize: 13,
                cursor: "pointer",
                transition: "all 0.15s",
              }}
              onMouseOver={(e) => (e.currentTarget.style.background = "rgba(16,185,129,0.18)")}
              onMouseOut={(e) => (e.currentTarget.style.background = "rgba(16,185,129,0.10)")}
            >
              ✓ Submit Exam
            </button>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes slideInDown {
          from { transform: translateY(-100%); }
          to   { transform: translateY(0); }
        }
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
