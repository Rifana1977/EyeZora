"use client";

import { useEffect, useState } from "react";
import Sidebar from "@/app/components/Sidebar";
import { studentApi } from "@/lib/api";
import { toast } from "@/app/components/ui/Toast";
import { SkeletonRow } from "@/app/components/ui/Skeleton";

interface StudentResult {
  _id: string;
  examId: string;
  examTitle: string;
  score: number;
  totalMarks: number;
  percentage: number;
  submittedAt: string;
}

export default function StudentResultsPage() {
  const [results, setResults] = useState<StudentResult[]>([]);
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    try {
      const data = await studentApi.getMyResults();
      setResults(data);
    } catch (err: any) {
      toast.error(err.message || "Failed to load results");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <div style={{ display: "flex", minHeight: "100vh" }} className="animated-bg">
      <Sidebar role="student" />

      <main style={{ flex: 1, padding: "36px 40px", overflowY: "auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 28 }}>
          <h1 style={{ color: "var(--text-primary)", fontSize: 26, fontWeight: 800, marginBottom: 4 }}>
            My Published Exam Results
          </h1>
          <p style={{ color: "var(--text-muted)", fontSize: 14 }}>
            View your scores and details for completed examinations.
          </p>
        </div>

        {/* Results Container */}
        <div className="glass-card" style={{ padding: 24 }}>
          {loading ? (
            <div style={{ padding: "40px 0", textAlign: "center" }}>
              <div style={{
                width: 32, height: 32, border: "3px solid rgba(124,58,237,0.3)",
                borderTopColor: "#7c3aed", borderRadius: "50%",
                animation: "spin 0.8s linear infinite", margin: "0 auto 12px"
              }} />
              <p style={{ color: "var(--text-secondary)", fontSize: 14 }}>Loading results…</p>
              <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
            </div>
          ) : results.length === 0 ? (
            <div style={{ textAlign: "center", padding: "60px 0", color: "var(--text-muted)" }}>
              <div style={{ fontSize: 44, marginBottom: 12 }}>🔒</div>
              <p style={{ fontSize: 15, fontWeight: 600, margin: 0 }}>No published results available</p>
              <p style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 6 }}>
                Once your exam results are graded and published by the instructor, they will appear here.
              </p>
            </div>
          ) : (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: 16 }}>
              {results.map((r) => (
                <div
                  key={r._id}
                  style={{
                    padding: 20, borderRadius: 12, border: "1px solid var(--bg-border)",
                    background: "var(--bg-elevated)", transition: "all 0.15s"
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14 }}>
                    <div>
                      <h3 style={{ color: "var(--text-primary)", fontSize: 16, fontWeight: 700, margin: 0 }}>
                        {r.examTitle}
                      </h3>
                      <span style={{ color: "var(--text-muted)", fontSize: 11 }}>
                        Submitted: {new Date(r.submittedAt).toLocaleDateString("en-IN")}
                      </span>
                    </div>
                    <span style={{
                      padding: "4px 8px", borderRadius: 6, fontSize: 12, fontWeight: 700,
                      background: r.percentage >= 50 ? "rgba(16,185,129,0.1)" : "rgba(239,68,68,0.08)",
                      color: r.percentage >= 50 ? "#10b981" : "#ef4444"
                    }}>
                      {r.percentage}%
                    </span>
                  </div>

                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingTop: 12, borderTop: "1px solid var(--bg-border)" }}>
                    <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>Final Score</span>
                    <strong style={{ color: "var(--text-primary)", fontSize: 16 }}>
                      {r.score} <span style={{ color: "var(--text-muted)", fontSize: 12, fontWeight: 400 }}>/ {r.totalMarks}</span>
                    </strong>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
