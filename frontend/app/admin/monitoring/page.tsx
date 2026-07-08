"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import Sidebar from "@/app/components/Sidebar";
import { adminApi } from "@/lib/api";
import { RiskBadge, StatusBadge } from "@/app/components/ui/Badge";
import { SkeletonRow } from "@/app/components/ui/Skeleton";
import { toast } from "@/app/components/ui/Toast";
import { ConfirmDialog } from "@/app/components/ui/Modal";
import { useRowSelection } from "@/lib/hooks/useRowSelection";
import { SelectAllCheckbox } from "@/app/components/ui/SelectAllCheckbox";
import { BulkActionToolbar } from "@/app/components/ui/BulkActionToolbar";

interface Session {
  _id: string;
  studentId: string;
  studentName: string;
  examTitle: string;
  startTime: string;
  endTime: string;
  durationSeconds: number;
  status: string;
  totalViolations: number;
  tabSwitchCount: number;
  fullscreenExitCount: number;
  riskLevel: "Low" | "Medium" | "High";
  recordingUrl: string | null;
}

export default function MonitoringPage() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<"all" | "Low" | "Medium" | "High">("all");

  // Bulk selection
  const sel = useRowSelection();
  const [bulkDeleteOpen, setBulkDeleteOpen] = useState(false);
  const [bulkDeleting, setBulkDeleting] = useState(false);

  useEffect(() => {
    function fetchSessions() {
      adminApi.getSessions()
        .then(setSessions)
        .catch((e: any) => toast.error(e.message))
        .finally(() => setLoading(false));
    }

    fetchSessions();
    const interval = setInterval(fetchSessions, 3000);
    return () => clearInterval(interval);
  }, []);

  const filtered = sessions.filter((s) => {
    const matchSearch =
      s.studentName.toLowerCase().includes(search.toLowerCase()) ||
      s.studentId.toLowerCase().includes(search.toLowerCase());
    const matchFilter = filter === "all" || s.riskLevel === filter;
    return matchSearch && matchFilter;
  });

  const filteredIds = filtered.map((s) => s._id);

  function formatDuration(sec: number) {
    if (!sec) return "—";
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m}m ${s}s`;
  }

  async function handleBulkDelete() {
    setBulkDeleting(true);
    try {
      const res = await adminApi.bulkDeleteSessions(sel.selectedArray);
      const { deleted, requested } = res;
      if (deleted === requested) {
        toast.success(`${deleted} session${deleted !== 1 ? "s" : ""} deleted successfully.`);
      } else {
        toast.success(`Deleted ${deleted} of ${requested} sessions.`);
      }
      sel.clearSelection();
      setBulkDeleteOpen(false);
      // Refresh
      adminApi.getSessions().then(setSessions).catch(() => {});
    } catch (err: any) {
      toast.error(err.message || "Bulk delete failed");
    } finally {
      setBulkDeleting(false);
    }
  }

  return (
    <div style={{ display: "flex", minHeight: "100vh" }} className="animated-bg">
      <Sidebar role="admin" />

      <main style={{ flex: 1, padding: "36px 40px", overflowY: "auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 28 }}>
          <h1 style={{ color: "var(--text-primary)", fontSize: 26, fontWeight: 800, marginBottom: 4 }}>
            Monitoring Dashboard
          </h1>
          <p style={{ color: "var(--text-muted)", fontSize: 14 }}>
            Review completed exam sessions and proctoring reports
          </p>
        </div>

        {/* Summary Cards */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16, marginBottom: 28 }}>
          <SummaryCard
            label="Total Sessions" value={sessions.length}
            color="#7c3aed" icon="📋"
          />
          <SummaryCard
            label="Flagged (High Risk)" value={sessions.filter((s) => s.riskLevel === "High").length}
            color="#ef4444" icon="🚨"
          />
          <SummaryCard
            label="Clean (Low Risk)" value={sessions.filter((s) => s.riskLevel === "Low").length}
            color="#10b981" icon="✅"
          />
        </div>

        {/* Filters */}
        <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap" }}>
          <input
            className="ez-input"
            placeholder="Search student name or ID…"
            value={search}
            onChange={(e) => { setSearch(e.target.value); sel.clearSelection(); }}
            style={{ maxWidth: 320 }}
          />
          <div style={{ display: "flex", gap: 8 }}>
            {(["all", "Low", "Medium", "High"] as const).map((f) => (
              <button
                key={f}
                onClick={() => { setFilter(f); sel.clearSelection(); }}
                style={{
                  padding: "9px 16px",
                  borderRadius: 8,
                  border: filter === f ? "1px solid rgba(124,58,237,0.5)" : "1px solid var(--bg-border)",
                  background: filter === f ? "rgba(124,58,237,0.15)" : "rgba(255,255,255,0.04)",
                  color: filter === f ? "#a78bfa" : "var(--text-secondary)",
                  cursor: "pointer",
                  fontSize: 13,
                  fontWeight: filter === f ? 700 : 500,
                  transition: "all 0.15s",
                }}
              >
                {f === "all" ? "All" : f}
              </button>
            ))}
          </div>
        </div>

        {/* Bulk Action Toolbar */}
        <BulkActionToolbar
          count={sel.selectedCount}
          onDelete={() => setBulkDeleteOpen(true)}
          onClear={sel.clearSelection}
          deleting={bulkDeleting}
          noun="sessions"
        />

        {/* Table */}
        <div className="glass-card" style={{ overflow: "hidden" }}>
          <div style={{ overflowX: "auto" }}>
            <table className="ez-table">
              <thead>
                <tr>
                  <th style={{ width: 40, textAlign: "center" }}>
                    <SelectAllCheckbox
                      checked={sel.isAllSelected(filteredIds)}
                      indeterminate={sel.isIndeterminate(filteredIds)}
                      onChange={() => sel.toggleAll(filteredIds)}
                      disabled={loading || filtered.length === 0}
                    />
                  </th>
                  <th>Student</th>
                  <th>Exam</th>
                  <th>Duration</th>
                  <th>Violations</th>
                  <th>Tab Switches</th>
                  <th>FS Exits</th>
                  <th>Risk</th>
                  <th>Recording</th>
                  <th>Report</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  [1, 2, 3, 4].map((i) => <SkeletonRow key={i} />)
                ) : filtered.length === 0 ? (
                  <tr>
                    <td colSpan={11} style={{ textAlign: "center", padding: "60px 0", color: "var(--text-muted)" }}>
                      <div style={{ fontSize: 40, marginBottom: 10 }}>📊</div>
                      <p>No sessions found</p>
                    </td>
                  </tr>
                ) : (
                  filtered.map((s) => {
                    const isSelected = sel.selected.has(s._id);
                    return (
                      <tr
                        key={s._id}
                        style={{
                          background: isSelected ? "rgba(124,58,237,0.06)" : undefined,
                          transition: "background 0.15s",
                        }}
                      >
                        <td style={{ textAlign: "center" }}>
                          <input
                            type="checkbox"
                            checked={isSelected}
                            onChange={() => sel.toggleOne(s._id)}
                            style={{ accentColor: "#7c3aed", width: 15, height: 15, cursor: "pointer" }}
                          />
                        </td>
                        <td>
                          <div style={{ fontWeight: 600, color: "var(--text-primary)", fontSize: 13 }}>{s.studentName}</div>
                          <div style={{ color: "#a78bfa", fontSize: 11, fontFamily: "monospace" }}>{s.studentId}</div>
                        </td>
                        <td style={{ maxWidth: 160 }}>
                          <span style={{ fontSize: 13, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
                            {s.examTitle}
                          </span>
                        </td>
                        <td>{formatDuration(s.durationSeconds)}</td>
                        <td>
                          <span style={{
                            fontWeight: 700,
                            color: s.totalViolations >= 6 ? "#ef4444" : s.totalViolations >= 3 ? "#f59e0b" : "#10b981",
                            fontSize: 15,
                          }}>
                            {s.totalViolations}
                          </span>
                        </td>
                        <td>{s.tabSwitchCount}</td>
                        <td>{s.fullscreenExitCount}</td>
                        <td><RiskBadge level={s.riskLevel} /></td>
                        <td>
                          {s.recordingUrl ? (
                            <a
                              href={s.recordingUrl}
                              target="_blank"
                              rel="noreferrer"
                              style={{
                                display: "inline-flex",
                                alignItems: "center",
                                gap: 6,
                                padding: "5px 12px",
                                borderRadius: 8,
                                background: "rgba(59,130,246,0.1)",
                                border: "1px solid rgba(59,130,246,0.3)",
                                color: "#3b82f6",
                                fontSize: 12,
                                fontWeight: 600,
                                textDecoration: "none",
                              }}
                            >
                              ▶ Play
                            </a>
                          ) : (
                            <span style={{ color: "var(--text-muted)", fontSize: 12 }}>—</span>
                          )}
                        </td>
                        <td>
                          <Link
                            href={`/admin/reports/${s._id}`}
                            style={{
                              display: "inline-flex",
                              alignItems: "center",
                              gap: 6,
                              padding: "5px 12px",
                              borderRadius: 8,
                              background: "rgba(124,58,237,0.1)",
                              border: "1px solid rgba(124,58,237,0.3)",
                              color: "#a78bfa",
                              fontSize: 12,
                              fontWeight: 600,
                              textDecoration: "none",
                            }}
                          >
                            📄 Report
                          </Link>
                        </td>
                        <td><StatusBadge status={s.status} /></td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>

      {/* Bulk Delete Confirm */}
      <ConfirmDialog
        isOpen={bulkDeleteOpen}
        onClose={() => setBulkDeleteOpen(false)}
        onConfirm={handleBulkDelete}
        title="Delete Session Records"
        message={`Are you sure you want to delete ${sel.selectedCount} session record${sel.selectedCount !== 1 ? "s" : ""}? Proctoring data will be permanently removed. This action cannot be undone.`}
        confirmLabel={`Delete ${sel.selectedCount} Session${sel.selectedCount !== 1 ? "s" : ""}`}
        danger
      />
    </div>
  );
}

function SummaryCard({ label, value, color, icon }: { label: string; value: number; color: string; icon: string }) {
  return (
    <div style={{
      background: "var(--bg-card)",
      border: "1px solid var(--bg-border)",
      borderRadius: 16,
      padding: "20px 24px",
      display: "flex",
      alignItems: "center",
      gap: 16,
    }}>
      <div style={{
        width: 48, height: 48, borderRadius: 12,
        background: `${color}15`,
        border: `1px solid ${color}30`,
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 22,
      }}>
        {icon}
      </div>
      <div>
        <p style={{ color: "var(--text-muted)", fontSize: 12, fontWeight: 600, marginBottom: 4 }}>{label}</p>
        <p style={{ color: "var(--text-primary)", fontSize: 28, fontWeight: 800 }}>{value}</p>
      </div>
    </div>
  );
}
