"use client";

import { useEffect, useState } from "react";
import Sidebar from "@/app/components/Sidebar";
import { adminApi } from "@/lib/api";
import { toast } from "@/app/components/ui/Toast";
import { SkeletonRow } from "@/app/components/ui/Skeleton";

interface Result {
  _id: string;
  studentId: string;
  studentName: string;
  studentEmail: string;
  examId: string;
  examTitle: string;
  score: number;
  totalMarks: number;
  percentage: number;
  resultsPublished: boolean;
  submittedAt: string;
}

interface Exam {
  _id: string;
  title: string;
}

export default function ReportsPage() {
  const [results, setResults] = useState<Result[]>([]);
  const [exams, setExams] = useState<Exam[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterExam, setFilterExam] = useState("all");
  const [search, setSearch] = useState("");
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [toggling, setToggling] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    try {
      const examId = filterExam !== "all" ? filterExam : undefined;
      const [resData, examsData] = await Promise.all([
        adminApi.getResults(examId),
        adminApi.getAllExams(),
      ]);
      setResults(resData);
      setExams(examsData);
    } catch (err: any) {
      toast.error(err.message || "Failed to load report data");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, [filterExam]);

  async function togglePublishSingle(id: string, currentlyPublished: boolean) {
    setToggling(id);
    try {
      if (currentlyPublished) {
        await adminApi.unpublishResults({ submissionIds: [id] });
        toast.success("Result unpublished");
      } else {
        const res = await adminApi.publishResults({ submissionIds: [id] });
        toast.success(res.message || "Result published successfully");
      }
      load();
    } catch (err: any) {
      toast.error(err.message || "Action failed");
    } finally {
      setToggling(null);
    }
  }

  async function handleBulkPublish(publish: boolean) {
    if (selectedIds.length === 0) {
      toast.error("No results selected");
      return;
    }
    try {
      if (publish) {
        const res = await adminApi.publishResults({ submissionIds: selectedIds });
        toast.success(res.message || `Published ${selectedIds.length} results`);
      } else {
        await adminApi.unpublishResults({ submissionIds: selectedIds });
        toast.success(`Unpublished ${selectedIds.length} results`);
      }
      setSelectedIds([]);
      load();
    } catch (err: any) {
      toast.error(err.message || "Action failed");
    }
  }

  const filtered = results.filter((r) =>
    r.studentName.toLowerCase().includes(search.toLowerCase()) ||
    r.studentId.toLowerCase().includes(search.toLowerCase()) ||
    r.examTitle.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div style={{ display: "flex", minHeight: "100vh" }} className="animated-bg">
      <Sidebar role="admin" />

      <main style={{ flex: 1, padding: "36px 40px", overflowY: "auto" }}>
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 28 }}>
          <div>
            <h1 style={{ color: "var(--text-primary)", fontSize: 26, fontWeight: 800, marginBottom: 4 }}>
              Exam Results & Publication Management
            </h1>
            <p style={{ color: "var(--text-muted)", fontSize: 14 }}>
              Review scores and toggle student access visibility for exam results.
            </p>
          </div>

          <div style={{ display: "flex", gap: 10 }}>
            {selectedIds.length > 0 && (
              <>
                <button
                  onClick={() => handleBulkPublish(true)}
                  style={{
                    padding: "9px 16px", borderRadius: 8, fontSize: 13, fontWeight: 600,
                    background: "rgba(16,185,129,0.1)", color: "#10b981", border: "1px solid rgba(16,185,129,0.3)",
                    cursor: "pointer"
                  }}
                >
                  📢 Publish Selected ({selectedIds.length})
                </button>
                <button
                  onClick={() => handleBulkPublish(false)}
                  style={{
                    padding: "9px 16px", borderRadius: 8, fontSize: 13, fontWeight: 600,
                    background: "rgba(239,68,68,0.08)", color: "#ef4444", border: "1px solid rgba(239,68,68,0.25)",
                    cursor: "pointer"
                  }}
                >
                  🔒 Unpublish Selected
                </button>
              </>
            )}
          </div>
        </div>

        {/* Filters */}
        <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap", alignItems: "center" }}>
          <input
            className="ez-input"
            placeholder="Search by student name, ID, or exam title…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ maxWidth: 360 }}
          />

          <select
            className="ez-input"
            value={filterExam}
            onChange={(e) => setFilterExam(e.target.value)}
            style={{ maxWidth: 220 }}
          >
            <option value="all">All Exams</option>
            {exams.map((ex) => (
              <option key={ex._id} value={ex._id}>
                {ex.title}
              </option>
            ))}
          </select>
        </div>

        {/* Results Table */}
        <div className="glass-card" style={{ overflow: "hidden" }}>
          <div style={{ overflowX: "auto" }}>
            <table className="ez-table">
              <thead>
                <tr>
                  <th style={{ width: 40 }}>
                    <input
                      type="checkbox"
                      checked={filtered.length > 0 && selectedIds.length === filtered.length}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedIds(filtered.map((r) => r._id));
                        } else {
                          setSelectedIds([]);
                        }
                      }}
                      style={{ accentColor: "#7c3aed" }}
                    />
                  </th>
                  <th>Student</th>
                  <th>Exam</th>
                  <th>Score</th>
                  <th>Percentage</th>
                  <th>Visibility</th>
                  <th>Submitted At</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  [1, 2, 3, 4].map((i) => <SkeletonRow key={i} />)
                ) : filtered.length === 0 ? (
                  <tr>
                    <td colSpan={8} style={{ textAlign: "center", padding: "60px 0", color: "var(--text-muted)" }}>
                      <div style={{ fontSize: 40, marginBottom: 10 }}>📊</div>
                      <p>No results found</p>
                    </td>
                  </tr>
                ) : (
                  filtered.map((r) => {
                    const isSelected = selectedIds.includes(r._id);
                    return (
                      <tr key={r._id}>
                        <td>
                          <input
                            type="checkbox"
                            checked={isSelected}
                            onChange={() => {
                              if (isSelected) {
                                setSelectedIds(selectedIds.filter((id) => id !== r._id));
                              } else {
                                setSelectedIds([...selectedIds, r._id]);
                              }
                            }}
                            style={{ accentColor: "#7c3aed" }}
                          />
                        </td>
                        <td>
                          <div style={{ fontWeight: 600, color: "var(--text-primary)" }}>{r.studentName}</div>
                          <div style={{ fontSize: 11, color: "#a78bfa", fontFamily: "monospace", marginTop: 2 }}>
                            {r.studentId}
                          </div>
                        </td>
                        <td>
                          <div style={{ fontWeight: 500, color: "var(--text-secondary)" }}>{r.examTitle}</div>
                        </td>
                        <td>
                          <strong style={{ color: "var(--text-primary)" }}>
                            {r.score} / {r.totalMarks}
                          </strong>
                        </td>
                        <td>
                          <span style={{
                            padding: "3px 8px", borderRadius: 6, fontSize: 12, fontWeight: 700,
                            background: r.percentage >= 50 ? "rgba(16,185,129,0.08)" : "rgba(239,68,68,0.06)",
                            color: r.percentage >= 50 ? "#10b981" : "#ef4444"
                          }}>
                            {r.percentage}%
                          </span>
                        </td>
                        <td>
                          <span style={{
                            padding: "3px 10px", borderRadius: 999, fontSize: 11, fontWeight: 700,
                            background: r.resultsPublished ? "rgba(16,185,129,0.12)" : "rgba(100,116,139,0.12)",
                            color: r.resultsPublished ? "#10b981" : "#64748b",
                            border: `1px solid ${r.resultsPublished ? "rgba(16,185,129,0.3)" : "rgba(100,116,139,0.3)"}`
                          }}>
                            {r.resultsPublished ? "📢 Published" : "🔒 Unpublished"}
                          </span>
                        </td>
                        <td style={{ fontSize: 12, color: "var(--text-muted)" }}>
                          {new Date(r.submittedAt).toLocaleString("en-IN", {
                            timeZone: "Asia/Kolkata",
                            day: "2-digit", month: "short",
                            hour: "2-digit", minute: "2-digit"
                          })}
                        </td>
                        <td>
                          <button
                            disabled={toggling === r._id}
                            onClick={() => togglePublishSingle(r._id, r.resultsPublished)}
                            style={{
                              padding: "6px 12px", borderRadius: 8, fontSize: 12, fontWeight: 600,
                              background: r.resultsPublished ? "rgba(239,68,68,0.08)" : "rgba(124,58,237,0.08)",
                              border: `1px solid ${r.resultsPublished ? "rgba(239,68,68,0.2)" : "rgba(124,58,237,0.2)"}`,
                              color: r.resultsPublished ? "#ef4444" : "#a78bfa",
                              cursor: "pointer", transition: "all 0.15s"
                            }}
                            onMouseOver={(e) => (e.currentTarget.style.background = r.resultsPublished ? "rgba(239,68,68,0.15)" : "rgba(124,58,237,0.15)")}
                            onMouseOut={(e) => (e.currentTarget.style.background = r.resultsPublished ? "rgba(239,68,68,0.08)" : "rgba(124,58,237,0.08)")}
                          >
                            {toggling === r._id ? "…" : r.resultsPublished ? "Unpublish" : "Publish"}
                          </button>
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
