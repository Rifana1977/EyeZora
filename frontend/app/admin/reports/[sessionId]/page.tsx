"use client";

import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import Link from "next/link";
import Sidebar from "@/app/components/Sidebar";
import { adminApi } from "@/lib/api";
import { RiskBadge, StatusBadge } from "@/app/components/ui/Badge";
import { Skeleton } from "@/app/components/ui/Skeleton";
import { toast } from "@/app/components/ui/Toast";

interface Log {
  _id: string;
  event: string;
  confidence: number;
  severity: "Low" | "Medium" | "High";
  timestamp: string;
}

interface Report {
  session: {
    _id: string;
    studentId: string;
    studentName: string;
    examTitle: string;
    startTime: string;
    endTime: string;
    durationSeconds: number;
    status: string;
    riskLevel: "Low" | "Medium" | "High";
    recordingUrl: string | null;
    audioRecordingUrl: string | null;
    logFilePath: string | null;
    score: number | null;
    totalMarks: number | null;
    multipleFacesCount: number;
    noFaceCount: number;
    lookingAwayCount: number;
    phoneDetectedCount: number;
    suspiciousMovementCount: number;
    tabSwitchCount: number;
    windowBlurCount: number;
    windowFocusLostCount: number;
    fullscreenExitCount: number;
    extensionWarningCount: number;
    totalViolations: number;
  };
  logs: Log[];
}

const EVENT_LABELS: Record<string, string> = {
  MULTIPLE_FACES:      "Multiple Faces Detected",
  NO_FACE:             "No Face Detected",
  LOOKING_AWAY:        "Looking Away",
  PHONE_DETECTED:      "Phone Detected",
  SUSPICIOUS_MOVEMENT: "Suspicious Movement",
  TAB_SWITCH:          "Browser Tab Switch",
  WINDOW_BLUR:         "Window Focus Lost",       // Legacy — display as focus lost
  WINDOW_FOCUS_LOST:   "Window Focus Lost",
  WINDOW_FOCUS_RESTORED: "Window Focus Restored",
  FULLSCREEN_EXIT:     "Fullscreen Exit",
  EXTENSION_WARNING:   "Extension Warning",
  CAMERA_DISCONNECTED: "Camera Disconnected",
  CAMERA_GRANTED:      "Camera Access Granted",
  MICROPHONE_DISABLED: "Microphone Disabled",
  EXAM_START:          "Exam Started",
  EXAM_END:            "Exam Ended",
};

const EVENT_ICONS: Record<string, string> = {
  MULTIPLE_FACES:      "👥",
  NO_FACE:             "🚫",
  LOOKING_AWAY:        "👀",
  PHONE_DETECTED:      "📱",
  SUSPICIOUS_MOVEMENT: "⚡",
  TAB_SWITCH:          "🔀",
  WINDOW_BLUR:         "🌫",
  WINDOW_FOCUS_LOST:   "🌫",
  WINDOW_FOCUS_RESTORED: "✅",
  FULLSCREEN_EXIT:     "⛶",
  EXTENSION_WARNING:   "🧩",
  CAMERA_DISCONNECTED: "📷",
  MICROPHONE_DISABLED: "🎙️",
  EXAM_START:          "▶",
  EXAM_END:            "⏹",
};

const SEVERITY_COLORS = {
  High:   { bg: "rgba(239,68,68,0.1)",  color: "#ef4444" },
  Medium: { bg: "rgba(245,158,11,0.1)", color: "#f59e0b" },
  Low:    { bg: "rgba(16,185,129,0.1)", color: "#10b981" },
};

export default function ReportPage() {
  const { sessionId } = useParams() as { sessionId: string };
  const [report, setReport] = useState<Report | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    adminApi.getSessionReport(sessionId)
      .then(setReport)
      .catch((e: any) => toast.error(e.message))
      .finally(() => setLoading(false));
  }, [sessionId]);

  function formatTime(t: string) {
    return new Date(t).toLocaleString("en-IN", { timeZone: "Asia/Kolkata" });
  }

  function formatTimeShort(t: string) {
    return new Date(t).toLocaleTimeString("en-IN", {
      timeZone: "Asia/Kolkata",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
  }

  function formatDuration(sec: number) {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m} min ${s} sec`;
  }

  if (loading) return (
    <div style={{ display: "flex", minHeight: "100vh" }} className="animated-bg">
      <Sidebar role="admin" />
      <main style={{ flex: 1, padding: "36px 40px" }}>
        <Skeleton height={32} width={300} borderRadius={10} />
        <div style={{ marginTop: 24 }}>
          {[1, 2, 3].map((i) => (
            <div key={i} style={{ marginBottom: 16 }}>
              <Skeleton height={80} borderRadius={12} />
            </div>
          ))}
        </div>
      </main>
    </div>
  );

  if (!report) return (
    <div style={{ display: "flex", minHeight: "100vh" }} className="animated-bg">
      <Sidebar role="admin" />
      <main style={{ flex: 1, padding: "36px 40px", color: "#ef4444" }}>
        Session not found.
      </main>
    </div>
  );

  const { session, logs } = report;

  const violationCategories = [
    { label: "Multiple Faces",     count: session.multipleFacesCount,      icon: "👥", color: "#ef4444" },
    { label: "No Face Detected",   count: session.noFaceCount,             icon: "🚫", color: "#ef4444" },
    { label: "Looking Away",       count: session.lookingAwayCount,        icon: "👀", color: "#f59e0b" },
    { label: "Phone Detected",     count: session.phoneDetectedCount,      icon: "📱", color: "#ef4444" },
    { label: "Suspicious Movement",count: session.suspiciousMovementCount, icon: "⚡", color: "#f59e0b" },
    { label: "Tab Switches",       count: session.tabSwitchCount,          icon: "🔀", color: "#ef4444" },
    { label: "Window Focus Lost",  count: (session.windowFocusLostCount || 0) + (session.windowBlurCount || 0), icon: "🌫", color: "#f59e0b" },
    { label: "Fullscreen Exits",   count: session.fullscreenExitCount,     icon: "⛶",  color: "#ef4444" },
    { label: "Extension Warnings", count: session.extensionWarningCount,   icon: "🧩", color: "#f59e0b" },
  ];

  const recommendation =
    session.riskLevel === "High"
      ? "⚠️ This exam should be manually reviewed. High risk of malpractice detected."
      : session.riskLevel === "Medium"
      ? "📋 Moderate violations detected. Review the event log for context."
      : "✅ Low violation count. Exam appears clean.";

  // Filter out informational events for violation count display
  const violationLogs = logs.filter(l => !["EXAM_START", "EXAM_END", "CAMERA_GRANTED", "WINDOW_FOCUS_RESTORED"].includes(l.event));

  return (
    <div style={{ display: "flex", minHeight: "100vh" }} className="animated-bg">
      <Sidebar role="admin" />

      <main style={{ flex: 1, padding: "36px 40px", overflowY: "auto" }}>
        {/* Breadcrumb */}
        <div style={{ marginBottom: 24, display: "flex", alignItems: "center", gap: 8, color: "var(--text-muted)", fontSize: 13 }}>
          <Link href="/admin/monitoring" style={{ color: "#a78bfa", textDecoration: "none" }}>
            ← Monitoring
          </Link>
          <span>/</span>
          <span>Report</span>
        </div>

        {/* Title */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 28 }}>
          <div>
            <h1 style={{ color: "var(--text-primary)", fontSize: 24, fontWeight: 800, marginBottom: 4 }}>
              Exam Report
            </h1>
            <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
              Session ID: <span style={{ fontFamily: "monospace" }}>{session._id}</span>
            </p>
          </div>
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <RiskBadge level={session.riskLevel} />
            <StatusBadge status={session.status} />
          </div>
        </div>

        {/* Student + Exam Info */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 24 }}>
          <InfoCard title="Student Details">
            <InfoRow label="Name" value={session.studentName} />
            <InfoRow label="Student ID" value={session.studentId} mono />
            <InfoRow label="Score" value={session.score !== null ? `${session.score} / ${session.totalMarks}` : "—"} />
          </InfoCard>
          <InfoCard title="Exam Details">
            <InfoRow label="Exam" value={session.examTitle} />
            <InfoRow label="Start" value={formatTime(session.startTime)} />
            <InfoRow label="End" value={session.endTime ? formatTime(session.endTime) : "—"} />
            <InfoRow label="Duration" value={session.durationSeconds ? formatDuration(session.durationSeconds) : "—"} />
          </InfoCard>
        </div>

        {/* Violation Summary */}
        <div className="glass-card" style={{ padding: 24, marginBottom: 24 }}>
          <h2 style={{ color: "var(--text-primary)", fontSize: 17, fontWeight: 700, marginBottom: 16 }}>
            🚨 Violation Summary
          </h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(150px,1fr))", gap: 12, marginBottom: 16 }}>
            {violationCategories.map((v) => (
              <div key={v.label} style={{
                background: v.count > 0 ? `${v.color}0D` : "rgba(255,255,255,0.03)",
                border: `1px solid ${v.count > 0 ? v.color + "30" : "var(--bg-border)"}`,
                borderRadius: 12,
                padding: "14px 16px",
              }}>
                <div style={{ fontSize: 22, marginBottom: 6 }}>{v.icon}</div>
                <div style={{ color: v.count > 0 ? v.color : "var(--text-muted)", fontSize: 22, fontWeight: 800 }}>
                  {v.count}
                </div>
                <div style={{ color: "var(--text-muted)", fontSize: 11, marginTop: 2 }}>{v.label}</div>
              </div>
            ))}
          </div>
          <div style={{
            padding: "14px 18px",
            borderRadius: 10,
            background: session.riskLevel === "High"
              ? "rgba(239,68,68,0.08)"
              : session.riskLevel === "Medium"
              ? "rgba(245,158,11,0.08)"
              : "rgba(16,185,129,0.08)",
            border: `1px solid ${session.riskLevel === "High" ? "rgba(239,68,68,0.3)" : session.riskLevel === "Medium" ? "rgba(245,158,11,0.3)" : "rgba(16,185,129,0.3)"}`,
          }}>
            <p style={{ color: "var(--text-primary)", fontSize: 14, fontWeight: 600, margin: 0 }}>
              {recommendation}
            </p>
          </div>
        </div>

        {/* Video Recording */}
        {session.recordingUrl && (
          <div className="glass-card" style={{ padding: 24, marginBottom: 24 }}>
            <h2 style={{ color: "var(--text-primary)", fontSize: 17, fontWeight: 700, marginBottom: 16 }}>
              🎬 Video Recording
            </h2>
            <video
              src={session.recordingUrl}
              controls
              style={{ width: "100%", maxHeight: 400, borderRadius: 12, background: "#000" }}
            />
          </div>
        )}

        {/* Audio Recording */}
        {session.audioRecordingUrl && (
          <div className="glass-card" style={{ padding: 24, marginBottom: 24 }}>
            <h2 style={{ color: "var(--text-primary)", fontSize: 17, fontWeight: 700, marginBottom: 16 }}>
              🎙️ Audio Recording
            </h2>
            <div style={{
              background: "var(--bg-elevated)", border: "1px solid var(--bg-border)",
              borderRadius: 12, padding: "20px 24px",
            }}>
              <p style={{ color: "var(--text-muted)", fontSize: 12, marginBottom: 12 }}>
                Microphone recording captured during examination
              </p>
              <audio
                src={session.audioRecordingUrl}
                controls
                style={{ width: "100%", accentColor: "#7c3aed" }}
              />
            </div>
          </div>
        )}

        {/* Event Log */}
        <div className="glass-card" style={{ padding: 24 }}>
          <h2 style={{ color: "var(--text-primary)", fontSize: 17, fontWeight: 700, marginBottom: 16 }}>
            📋 Event Log ({violationLogs.length} violations)
          </h2>
          <div style={{ overflowX: "auto" }}>
            <table className="ez-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Event</th>
                  <th>Confidence</th>
                  <th>Severity</th>
                </tr>
              </thead>
              <tbody>
                {logs.length === 0 ? (
                  <tr>
                    <td colSpan={4} style={{ textAlign: "center", color: "var(--text-muted)", padding: "40px 0" }}>
                      No events recorded
                    </td>
                  </tr>
                ) : (
                  logs.map((log) => {
                    const sc = SEVERITY_COLORS[log.severity] || SEVERITY_COLORS.Low;
                    const isInfo = ["EXAM_START", "EXAM_END", "CAMERA_GRANTED", "WINDOW_FOCUS_RESTORED"].includes(log.event);
                    return (
                      <tr key={log._id}>
                        <td style={{ fontFamily: "monospace", fontSize: 12, whiteSpace: "nowrap" }}>
                          {formatTimeShort(log.timestamp)}
                        </td>
                        <td>
                          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <span style={{ fontSize: 14 }}>{EVENT_ICONS[log.event] || "•"}</span>
                            <span style={{
                              color: isInfo ? "var(--text-secondary)" : "var(--text-primary)",
                              fontWeight: isInfo ? 400 : 600,
                              fontSize: 13,
                            }}>
                              {EVENT_LABELS[log.event] || log.event}
                            </span>
                          </div>
                        </td>
                        <td>
                          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <div style={{
                              height: 6, width: 60, borderRadius: 3,
                              background: "rgba(255,255,255,0.08)",
                              overflow: "hidden",
                            }}>
                              <div style={{
                                height: "100%",
                                width: `${log.confidence}%`,
                                background: sc.color,
                                borderRadius: 3,
                              }} />
                            </div>
                            <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>
                              {log.confidence}%
                            </span>
                          </div>
                        </td>
                        <td>
                          <span style={{
                            padding: "3px 10px",
                            borderRadius: 999,
                            fontSize: 11,
                            fontWeight: 700,
                            background: sc.bg,
                            color: sc.color,
                          }}>
                            {log.severity}
                          </span>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}

function InfoCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="glass-card" style={{ padding: 22 }}>
      <h3 style={{ color: "#a78bfa", fontSize: 12, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 16 }}>
        {title}
      </h3>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {children}
      </div>
    </div>
  );
}

function InfoRow({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
      <span style={{ color: "var(--text-muted)", fontSize: 13, flexShrink: 0 }}>{label}</span>
      <span style={{
        color: "var(--text-primary)",
        fontSize: 13,
        fontWeight: 600,
        fontFamily: mono ? "monospace" : "inherit",
        textAlign: "right",
      }}>
        {value}
      </span>
    </div>
  );
}
