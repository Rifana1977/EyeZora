"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { getUser, clearAuth, type AssignedExam } from "@/lib/auth";
import { studentApi } from "@/lib/api";

type PreExamState = "loading" | "no-assignment" | "expired" | "upcoming" | "active";

interface PermissionState {
  camera: "pending" | "granted" | "denied";
  microphone: "pending" | "granted" | "denied";
}

export default function PreExamPage() {
  const router = useRouter();
  const user = getUser();
  const assignment = user?.assignedExam as AssignedExam | null | undefined;

  const [state, setState] = useState<PreExamState>("loading");
  const [questionCount, setQuestionCount] = useState<number | null>(null);
  const [countdown, setCountdown] = useState<string>("--:--:--");
  const [startDisabled, setStartDisabled] = useState(true);

  // Permission request flow
  const [requestingPerms, setRequestingPerms] = useState(false);
  const [permState, setPermState] = useState<PermissionState>({
    camera: "pending",
    microphone: "pending",
  });
  const [permError, setPermError] = useState<string | null>(null);

  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  function startCountdown(targetDate: Date) {
    function tick() {
      const diff = targetDate.getTime() - Date.now();
      if (diff <= 0) {
        setCountdown("00:00:00");
        setStartDisabled(false);
        setState("active");
        if (countdownRef.current) clearInterval(countdownRef.current);
        return;
      }
      const h = Math.floor(diff / 3600000);
      const m = Math.floor((diff % 3600000) / 60000);
      const s = Math.floor((diff % 60000) / 1000);
      setCountdown(
        `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
      );
    }
    tick();
    countdownRef.current = setInterval(tick, 1000);
  }

  // Determine which state to show
  useEffect(() => {
    if (!user) {
      router.replace("/student/login");
      return;
    }

    if (!assignment) {
      setTimeout(() => setState("no-assignment"), 0);
      return;
    }

    const now = new Date();
    const start = new Date(assignment.startTime);
    const end = new Date(assignment.endTime);

    if (now > end) {
      setTimeout(() => setState("expired"), 0);
      return;
    }

    if (now < start) {
      setTimeout(() => {
        setState("upcoming");
        startCountdown(start);
      }, 0);
      return;
    }

    // Active window — load question count
    setTimeout(() => {
      setState("active");
      setStartDisabled(false);
    }, 0);
    studentApi.getExamQuestions(assignment.id)
      .then((qs) => setQuestionCount(qs.length))
      .catch(() => setQuestionCount(null));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    return () => {
      if (countdownRef.current) clearInterval(countdownRef.current);
    };
  }, []);

  // ── Start Exam Flow ─────────────────────────────────────────────────────────

  const handleStartExam = useCallback(async () => {
    if (startDisabled || requestingPerms) return;
    setRequestingPerms(true);
    setPermError(null);

    // ── Step 1: Camera ──────────────────────────────────────────────────
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      setPermState(p => ({ ...p, camera: "granted" }));
    } catch {
      setPermState(p => ({ ...p, camera: "denied" }));
      setPermError("Camera permission is required. Please allow camera access and try again.");
      setRequestingPerms(false);
      return;
    }

    // ── Step 2: Microphone ──────────────────────────────────────────────
    try {
      const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      // Merge audio tracks into the same stream
      audioStream.getAudioTracks().forEach(t => stream.addTrack(t));
      setPermState(p => ({ ...p, microphone: "granted" }));
    } catch {
      setPermState(p => ({ ...p, microphone: "denied" }));
      setPermError("Microphone permission is required. Please allow microphone access and try again.");
      // Release camera
      stream.getTracks().forEach(t => t.stop());
      streamRef.current = null;
      setRequestingPerms(false);
      return;
    }

    // ── Step 3: Save stream and navigate ─────────────────────────────────
    // Pass stream via sessionStorage signal; actual stream object is held in ref
    // We store it globally so exam page can pick it up
    if (typeof window !== "undefined") {
      (window as Window & { __eyezoraStream?: MediaStream | null }).__eyezoraStream = streamRef.current;
      sessionStorage.setItem("ez_exam_ready", "true");
    }

    router.push("/student/exam");
  }, [startDisabled, requestingPerms, router]);

  function logout() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
    }
    clearAuth();
    document.cookie = "ez_token=; Max-Age=0; path=/";
    document.cookie = "ez_role=; Max-Age=0; path=/";
    router.replace("/student/login");
  }

  function formatDateTime(iso: string) {
    return new Date(iso).toLocaleString("en-IN", {
      timeZone: "Asia/Kolkata",
      weekday: "long", day: "numeric", month: "long",
      hour: "2-digit", minute: "2-digit",
    });
  }

  // ── Render States ───────────────────────────────────────────────────────────

  if (state === "loading") {
    return (
      <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "var(--bg-base)" }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ width: 48, height: 48, border: "3px solid rgba(124,58,237,0.3)", borderTopColor: "#7c3aed", borderRadius: "50%", animation: "spin 0.8s linear infinite", margin: "0 auto 16px" }} />
          <p style={{ color: "var(--text-muted)" }}>Loading your exam details…</p>
        </div>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  if (state === "no-assignment") {
    return (
      <div className="animated-bg" style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", padding: 20 }}>
        <div className="glass-card fade-in" style={{ padding: 48, textAlign: "center", maxWidth: 480 }}>
          <div style={{ fontSize: 56, marginBottom: 20 }}>📋</div>
          <h2 style={{ color: "var(--text-primary)", fontSize: 22, fontWeight: 800, marginBottom: 12 }}>
            No Exam Assigned
          </h2>
          <p style={{ color: "var(--text-muted)", fontSize: 14, lineHeight: 1.7, marginBottom: 28 }}>
            No exam has been assigned to you yet. Please contact your administrator to get an exam scheduled.
          </p>
          <div style={{
            padding: "14px 18px",
            background: "rgba(124,58,237,0.06)",
            border: "1px solid rgba(124,58,237,0.15)",
            borderRadius: 10,
            marginBottom: 28,
          }}>
            <p style={{ color: "var(--text-secondary)", fontSize: 13, margin: 0, lineHeight: 1.6 }}>
              Logged in as: <strong style={{ color: "var(--text-accent)" }}>{user?.name}</strong>
              {" "}({user?.studentId})
            </p>
          </div>
          <button onClick={logout} style={{
            padding: "11px 24px", borderRadius: 10, fontSize: 14, fontWeight: 600,
            background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.25)",
            color: "#ef4444", cursor: "pointer",
          }}>
            Logout
          </button>
        </div>
      </div>
    );
  }

  if (state === "expired") {
    return (
      <div className="animated-bg" style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", padding: 20 }}>
        <div className="glass-card fade-in" style={{ padding: 48, textAlign: "center", maxWidth: 480 }}>
          <div style={{ fontSize: 56, marginBottom: 20 }}>⏰</div>
          <h2 style={{ color: "#ef4444", fontSize: 22, fontWeight: 800, marginBottom: 12 }}>
            Exam Expired
          </h2>
          <p style={{ color: "var(--text-muted)", fontSize: 14, lineHeight: 1.7, marginBottom: 16 }}>
            This exam is no longer available. The scheduled examination window has passed.
          </p>
          {assignment && (
            <div style={{ padding: "14px 18px", background: "rgba(239,68,68,0.06)", border: "1px solid rgba(239,68,68,0.2)", borderRadius: 10, marginBottom: 28 }}>
              <p style={{ color: "var(--text-primary)", fontSize: 13, fontWeight: 600, marginBottom: 4 }}>
                {assignment.title}
              </p>
              <p style={{ color: "var(--text-muted)", fontSize: 12, margin: 0 }}>
                Was scheduled until: {formatDateTime(assignment.endTime)}
              </p>
            </div>
          )}
          <p style={{ color: "var(--text-muted)", fontSize: 13, marginBottom: 20 }}>
            Please contact your administrator to reschedule.
          </p>
          <button onClick={logout} style={{
            padding: "11px 24px", borderRadius: 10, fontSize: 14, fontWeight: 600,
            background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.25)",
            color: "#ef4444", cursor: "pointer",
          }}>
            Logout
          </button>
        </div>
      </div>
    );
  }

  if (state === "upcoming") {
    return (
      <div className="animated-bg" style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", padding: 20 }}>
        <div className="glass-card fade-in" style={{ padding: 48, maxWidth: 520, width: "100%" }}>
          {/* Brand */}
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 28 }}>
            <div style={{ width: 44, height: 44, borderRadius: 12, background: "linear-gradient(135deg,#7c3aed,#5b21b6)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 22 }}>👁</div>
            <div>
              <p style={{ color: "#a78bfa", fontSize: 11, fontWeight: 700, letterSpacing: "0.07em", textTransform: "uppercase", marginBottom: 2 }}>EYEZORA</p>
              <p style={{ color: "var(--text-muted)", fontSize: 12 }}>Exam Waiting Room</p>
            </div>
          </div>

          <h2 style={{ color: "var(--text-primary)", fontSize: 22, fontWeight: 800, marginBottom: 6 }}>
            {assignment?.title}
          </h2>
          <p style={{ color: "var(--text-muted)", fontSize: 13, marginBottom: 28 }}>
            Duration: {assignment?.duration} minutes
          </p>

          {/* Countdown */}
          <div style={{
            padding: "28px 24px",
            background: "rgba(124,58,237,0.06)",
            border: "1px solid rgba(124,58,237,0.2)",
            borderRadius: 16,
            textAlign: "center",
            marginBottom: 24,
          }}>
            <p style={{ color: "var(--text-muted)", fontSize: 12, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 10 }}>
              Exam Begins In
            </p>
            <div style={{ fontFamily: "monospace", fontSize: 48, fontWeight: 800, color: "#a78bfa", letterSpacing: 4 }}>
              {countdown}
            </div>
            <p style={{ color: "var(--text-muted)", fontSize: 12, marginTop: 10 }}>
              Your exam begins at {assignment ? formatDateTime(assignment.startTime) : ""}
            </p>
          </div>

          {/* Schedule details */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 24 }}>
            {[
              { label: "Start Time", value: assignment ? formatDateTime(assignment.startTime) : "—" },
              { label: "End Time", value: assignment ? formatDateTime(assignment.endTime) : "—" },
              { label: "Duration", value: `${assignment?.duration} minutes` },
              { label: "Questions", value: questionCount !== null ? `${questionCount} questions` : "Loading…" },
            ].map(item => (
              <div key={item.label} style={{
                padding: "12px 14px", borderRadius: 10,
                background: "var(--bg-elevated)", border: "1px solid var(--bg-border)",
              }}>
                <p style={{ color: "var(--text-muted)", fontSize: 11, fontWeight: 600, textTransform: "uppercase", marginBottom: 4 }}>{item.label}</p>
                <p style={{ color: "var(--text-primary)", fontSize: 13, fontWeight: 600, margin: 0 }}>{item.value}</p>
              </div>
            ))}
          </div>

          <button
            disabled={startDisabled}
            onClick={handleStartExam}
            className="btn-glow"
            style={{ width: "100%", padding: "14px 0", borderRadius: 14, fontSize: 15, fontWeight: 700, opacity: startDisabled ? 0.45 : 1 }}
          >
            {startDisabled ? `⏳ Waiting for exam to begin…` : "Start Exam →"}
          </button>

          <button onClick={logout} style={{
            width: "100%", marginTop: 10, padding: "10px 0", borderRadius: 10, fontSize: 13, fontWeight: 500,
            background: "transparent", border: "1px solid var(--bg-border)", color: "var(--text-muted)", cursor: "pointer",
          }}>
            Logout
          </button>
        </div>
      </div>
    );
  }

  // ── Active State: Pre-Exam Details Page ─────────────────────────────────────
  return (
    <div className="animated-bg" style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", padding: 20 }}>
      <div className="glass-card fade-in" style={{ padding: 40, maxWidth: 560, width: "100%" }}>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 28 }}>
          <div style={{ width: 52, height: 52, borderRadius: 14, background: "linear-gradient(135deg,#7c3aed,#5b21b6)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 26, boxShadow: "0 4px 20px rgba(124,58,237,0.4)" }}>
            👁
          </div>
          <div>
            <h1 style={{ color: "var(--text-primary)", fontSize: 20, fontWeight: 800, marginBottom: 2 }}>
              {assignment?.title || "Examination"}
            </h1>
            <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
              {assignment?.duration} minutes · {questionCount !== null ? `${questionCount} questions` : "Loading questions…"} · AI Proctored
            </p>
          </div>
        </div>

        {/* Student Info */}
        <div style={{ padding: "12px 16px", background: "rgba(124,58,237,0.06)", border: "1px solid rgba(124,58,237,0.15)", borderRadius: 10, marginBottom: 22 }}>
          <p style={{ color: "var(--text-muted)", fontSize: 11, marginBottom: 4 }}>Logged in as</p>
          <p style={{ color: "var(--text-primary)", fontWeight: 700, marginBottom: 2 }}>{user?.name}</p>
          <p style={{ color: "#a78bfa", fontSize: 12, fontFamily: "monospace" }}>{user?.studentId}</p>
        </div>

        {/* Exam Rules */}
        <div style={{ marginBottom: 22 }}>
          <h3 style={{ color: "var(--text-secondary)", fontSize: 12, fontWeight: 700, marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.07em" }}>
            Exam Rules & Guidelines
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
            {[
              { icon: "📷", text: "Your camera and microphone will be monitored throughout the exam" },
              { icon: "👤", text: "Only one face should be visible at all times" },
              { icon: "📵", text: "No phones, books, or additional devices are permitted" },
              { icon: "⛶", text: "Browser will enter full-screen mode — exiting is a violation" },
              { icon: "🔀", text: "Switching tabs or windows will be recorded as a violation" },
              { icon: "🤖", text: "AI monitors every second — all violations are logged and reported" },
            ].map((r) => (
              <div key={r.text} style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                <span style={{ fontSize: 15, flexShrink: 0, marginTop: 1 }}>{r.icon}</span>
                <p style={{ color: "var(--text-secondary)", fontSize: 13, margin: 0, lineHeight: 1.5 }}>{r.text}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Permissions Required */}
        <div style={{ marginBottom: 24 }}>
          <h3 style={{ color: "var(--text-secondary)", fontSize: 12, fontWeight: 700, marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.07em" }}>
            Permissions Required
          </h3>
          <div style={{ display: "flex", gap: 10 }}>
            {[
              { icon: "📷", label: "Camera", key: "camera" as const },
              { icon: "🎙️", label: "Microphone", key: "microphone" as const },
            ].map(p => {
              const ps = permState[p.key];
              return (
                <div key={p.key} style={{
                  flex: 1, padding: "12px 8px", borderRadius: 10, textAlign: "center",
                  background: ps === "granted" ? "rgba(16,185,129,0.08)" : ps === "denied" ? "rgba(239,68,68,0.08)" : "var(--bg-elevated)",
                  border: `1px solid ${ps === "granted" ? "rgba(16,185,129,0.25)" : ps === "denied" ? "rgba(239,68,68,0.25)" : "var(--bg-border)"}`,
                  transition: "all 0.2s",
                }}>
                  <div style={{ fontSize: 20, marginBottom: 4 }}>{p.icon}</div>
                  <div style={{ fontSize: 11, fontWeight: 600, color: ps === "granted" ? "#10b981" : ps === "denied" ? "#ef4444" : "var(--text-muted)" }}>
                    {p.label}
                  </div>
                  <div style={{ fontSize: 10, marginTop: 2, color: ps === "granted" ? "#10b981" : ps === "denied" ? "#ef4444" : "var(--text-muted)" }}>
                    {ps === "granted" ? "✓ Granted" : ps === "denied" ? "✗ Denied" : "Required"}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Permission Error */}
        {permError && (
          <div style={{
            padding: "12px 16px", borderRadius: 10, marginBottom: 16,
            background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.25)",
            animation: "fadeIn 0.3s ease",
          }}>
            <p style={{ color: "#ef4444", fontSize: 13, margin: 0, fontWeight: 500 }}>
              ⚠️ {permError}
            </p>
          </div>
        )}

        {/* Start Button */}
        <button
          id="start-exam-btn"
          onClick={handleStartExam}
          disabled={startDisabled || requestingPerms}
          className="btn-glow"
          style={{ width: "100%", padding: "15px 0", borderRadius: 14, fontSize: 16, fontWeight: 700 }}
        >
          {requestingPerms ? (
            <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 10 }}>
              <span style={{ width: 16, height: 16, border: "2px solid rgba(255,255,255,0.3)", borderTopColor: "#fff", borderRadius: "50%", display: "inline-block", animation: "spin 0.7s linear infinite" }} />
              Requesting Permissions…
            </span>
          ) : (
            "Start Exam — Allow Camera, Mic & Fullscreen"
          )}
        </button>

        <button onClick={logout} style={{
          width: "100%", marginTop: 10, padding: "10px 0", borderRadius: 10, fontSize: 13, fontWeight: 500,
          background: "transparent", border: "1px solid var(--bg-border)", color: "var(--text-muted)", cursor: "pointer",
        }}>
          Logout
        </button>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
