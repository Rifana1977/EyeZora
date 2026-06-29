"use client";

import { useEffect, useState } from "react";
import Sidebar from "@/app/components/Sidebar";
import { adminApi } from "@/lib/api";
import { SkeletonCard } from "@/app/components/ui/Skeleton";
import { RiskBadge } from "@/app/components/ui/Badge";

interface Stats {
  totalExams: number;
  totalStudents: number;
  completedSessions: number;
  flaggedSessions: number;
  recentSessions: any[];
}

export default function AdminDashboard() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    adminApi.getStats()
      .then(setStats)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  return (
    <div style={{ display: "flex", minHeight: "100vh" }} className="animated-bg">
      <Sidebar role="admin" />

      <main style={{ flex: 1, padding: "36px 40px", overflowY: "auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <h1 style={{ color: "var(--text-primary)", fontSize: 26, fontWeight: 800, marginBottom: 4 }}>
            Admin Dashboard
          </h1>
          <p style={{ color: "var(--text-muted)", fontSize: 14 }}>
            Overview of the EyeZora examination system
          </p>
        </div>

        {/* Stats Grid */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(200px,1fr))", gap: 20, marginBottom: 36 }}>
          {loading ? (
            [1, 2, 3, 4].map((i) => <SkeletonCard key={i} />)
          ) : (
            <>
              <StatCard title="Total Exams" value={stats?.totalExams ?? 0} icon="📋" color="#7c3aed" />
              <StatCard title="Students" value={stats?.totalStudents ?? 0} icon="👥" color="#3b82f6" />
              <StatCard title="Sessions" value={stats?.completedSessions ?? 0} icon="✅" color="#10b981" />
              <StatCard title="Flagged" value={stats?.flaggedSessions ?? 0} icon="🚨" color="#ef4444" />
            </>
          )}
        </div>

        {/* Recent Sessions */}
        <div className="glass-card" style={{ padding: 28 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
            <h2 style={{ color: "var(--text-primary)", fontSize: 18, fontWeight: 700 }}>
              Recent Exam Sessions
            </h2>
            <a href="/admin/monitoring" style={{
              color: "#a78bfa",
              textDecoration: "none",
              fontSize: 13,
              fontWeight: 600,
            }}>
              View All →
            </a>
          </div>

          {!stats?.recentSessions?.length ? (
            <div style={{ textAlign: "center", padding: "40px 0", color: "var(--text-muted)" }}>
              <div style={{ fontSize: 40, marginBottom: 10 }}>📊</div>
              <p>No exam sessions yet</p>
            </div>
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table className="ez-table">
                <thead>
                  <tr>
                    <th>Student</th>
                    <th>Exam</th>
                    <th>Status</th>
                    <th>Violations</th>
                    <th>Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {stats.recentSessions.map((s: any) => (
                    <tr key={s._id}>
                      <td>
                        <div style={{ fontWeight: 600, color: "var(--text-primary)" }}>{s.studentName}</div>
                        <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{s.studentId}</div>
                      </td>
                      <td>{s.examTitle}</td>
                      <td>
                        <span style={{
                          padding: "3px 10px",
                          borderRadius: 6,
                          fontSize: 12,
                          background: s.status === "completed"
                            ? "rgba(16,185,129,0.15)"
                            : s.status === "flagged"
                            ? "rgba(239,68,68,0.15)"
                            : "rgba(59,130,246,0.15)",
                          color: s.status === "completed" ? "#10b981"
                            : s.status === "flagged" ? "#ef4444"
                            : "#3b82f6",
                        }}>
                          {s.status}
                        </span>
                      </td>
                      <td>{s.totalViolations}</td>
                      <td><RiskBadge level={s.riskLevel} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* AI Status Card */}
        <div className="glass-card" style={{ padding: 24, marginTop: 24 }}>
          <h2 style={{ color: "var(--text-primary)", fontSize: 16, fontWeight: 700, marginBottom: 16 }}>
            AI Monitoring Status
          </h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))", gap: 12 }}>
            {[
              { name: "Face Detection", status: "Running" },
              { name: "Object Detection", status: "Running" },
              { name: "Gaze Analysis", status: "Active" },
              { name: "Proctoring Engine", status: "Online" },
            ].map((item) => (
              <div key={item.name} style={{
                background: "rgba(16,185,129,0.06)",
                border: "1px solid rgba(16,185,129,0.2)",
                borderRadius: 10,
                padding: "12px 16px",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}>
                <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>{item.name}</span>
                <span style={{ color: "#10b981", fontSize: 12, fontWeight: 700, display: "flex", alignItems: "center", gap: 6 }}>
                  <span className="pulse-dot" style={{ fontSize: 8 }}>●</span>
                  {item.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}

function StatCard({ title, value, icon, color }: { title: string; value: number; icon: string; color: string }) {
  return (
    <div className="stat-card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <p style={{ color: "var(--text-muted)", fontSize: 12, fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 8 }}>
            {title}
          </p>
          <p style={{ color: "var(--text-primary)", fontSize: 34, fontWeight: 800, lineHeight: 1 }}>
            {value}
          </p>
        </div>
        <div style={{
          width: 44,
          height: 44,
          borderRadius: 12,
          background: `${color}18`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 22,
          border: `1px solid ${color}30`,
        }}>
          {icon}
        </div>
      </div>
    </div>
  );
}
