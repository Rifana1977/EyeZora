"use client";

import { useEffect, useState, useCallback } from "react";
import Sidebar from "@/app/components/Sidebar";
import { adminApi } from "@/lib/api";
import Modal, { ConfirmDialog } from "@/app/components/ui/Modal";
import { toast } from "@/app/components/ui/Toast";
import { SkeletonRow } from "@/app/components/ui/Skeleton";

interface Assignment {
  _id: string;
  studentId: string;
  studentObjectId: { _id: string; studentId: string; name: string; email: string; isActive: boolean };
  examId: { _id: string; title: string; duration: number };
  startTime: string;
  endTime: string;
  duration: number;
  status: string;
  computedStatus: string;
  assignedAt: string;
  notes: string;
}

interface Student {
  _id: string;
  studentId: string;
  name: string;
  email: string;
  isActive: boolean;
}

interface Exam {
  _id: string;
  title: string;
  duration: number;
}

const STATUS_COLORS: Record<string, { bg: string; color: string; label: string }> = {
  assigned: { bg: "rgba(59,130,246,0.1)", color: "#3b82f6", label: "Assigned" },
  upcoming: { bg: "rgba(245,158,11,0.1)", color: "#f59e0b", label: "Upcoming" },
  active: { bg: "rgba(16,185,129,0.1)", color: "#10b981", label: "Active" },
  started: { bg: "rgba(16,185,129,0.12)", color: "#10b981", label: "In Progress" },
  completed: { bg: "rgba(107,114,128,0.1)", color: "#6b7280", label: "Completed" },
  expired: { bg: "rgba(239,68,68,0.08)", color: "#ef4444", label: "Expired" },
  cancelled: { bg: "rgba(107,114,128,0.08)", color: "#9ca3af", label: "Cancelled" },
};

function StatusBadge({ status }: { status: string }) {
  const s = STATUS_COLORS[status] || STATUS_COLORS.assigned;
  return (
    <span style={{
      padding: "3px 10px", borderRadius: 999, fontSize: 11, fontWeight: 700,
      background: s.bg, color: s.color, border: `1px solid ${s.color}30`,
    }}>
      {s.label}
    </span>
  );
}

const emptyForm = {
  studentObjectId: "",
  examId: "",
  startDate: "",
  startTime: "",
  duration: "",
  notes: "",
};

export default function AssignExamsPage() {
  const [assignments, setAssignments] = useState<Assignment[]>([]);
  const [students, setStudents] = useState<Student[]>([]);
  const [exams, setExams] = useState<Exam[]>([]);
  const [loading, setLoading] = useState(true);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const LIMIT = 15;

  const [search, setSearch] = useState("");
  const [filterExam, setFilterExam] = useState("all");
  const [filterStatus, setFilterStatus] = useState("all");

  const [modalOpen, setModalOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [form, setForm] = useState(emptyForm);
  const [saving, setSaving] = useState(false);

  const [cancelId, setCancelId] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const params: any = { page, limit: LIMIT };
      if (search) params.search = search;
      if (filterExam !== "all") params.examId = filterExam;
      if (filterStatus !== "all") params.status = filterStatus;

      const data = await adminApi.getAssignments(params);
      setAssignments(data.assignments || []);
      setTotal(data.total || 0);
      setTotalPages(data.totalPages || 1);
    } catch (err: any) {
      toast.error(err.message);
    } finally {
      setLoading(false);
    }
  }, [search, filterExam, filterStatus, page]);

  async function loadDropdowns() {
    try {
      const [s, e] = await Promise.all([adminApi.getStudents(), adminApi.getAllExams()]);
      setStudents(s);
      setExams(e);
    } catch { /* non-critical */ }
  }

  useEffect(() => { load(); }, [load]);
  useEffect(() => { loadDropdowns(); }, []);

  // Debounce search
  useEffect(() => {
    const t = setTimeout(() => { setPage(1); load(); }, 400);
    return () => clearTimeout(t);
  }, [search]); // eslint-disable-line react-hooks/exhaustive-deps

  function openAssign(existing?: Assignment) {
    if (existing) {
      setEditingId(existing._id);
      const st = new Date(existing.startTime);
      setForm({
        studentObjectId: existing.studentObjectId._id,
        examId: existing.examId._id,
        startDate: st.toISOString().slice(0, 10),
        startTime: st.toTimeString().slice(0, 5),
        duration: String(existing.duration),
        notes: existing.notes || "",
      });
    } else {
      setEditingId(null);
      setForm(emptyForm);
    }
    setModalOpen(true);
  }

  async function handleSave(e: React.FormEvent) {
    e.preventDefault();
    const { studentObjectId, examId, startDate, startTime, duration } = form;

    if (!studentObjectId || !examId || !startDate || !startTime || !duration) {
      toast.error("All fields except Notes are required");
      return;
    }

    const startDateTime = new Date(`${startDate}T${startTime}:00`);
    if (isNaN(startDateTime.getTime())) {
      toast.error("Invalid start date/time");
      return;
    }

    if (Number(duration) < 1) {
      toast.error("Duration must be at least 1 minute");
      return;
    }

    const endDateTime = new Date(startDateTime.getTime() + Number(duration) * 60 * 1000);

    setSaving(true);
    try {
      const payload = {
        studentObjectId,
        examId,
        startTime: startDateTime.toISOString(),
        endTime: endDateTime.toISOString(),
        duration: Number(duration),
        notes: form.notes,
      };

      if (editingId) {
        await adminApi.updateAssignment(editingId, payload);
        toast.success("Assignment updated successfully!");
      } else {
        await adminApi.createAssignment(payload);
        toast.success("Exam assigned successfully!");
      }
      setModalOpen(false);
      setForm(emptyForm);
      setEditingId(null);
      load();
    } catch (err: any) {
      toast.error(err.message);
    } finally {
      setSaving(false);
    }
  }

  async function handleCancel() {
    if (!cancelId) return;
    try {
      await adminApi.cancelAssignment(cancelId);
      toast.success("Assignment cancelled");
      setCancelId(null);
      load();
    } catch (err: any) {
      toast.error(err.message);
    }
  }

  function formatDateTime(iso: string) {
    if (!iso) return "—";
    return new Date(iso).toLocaleString("en-IN", {
      timeZone: "Asia/Kolkata",
      day: "2-digit", month: "short",
      hour: "2-digit", minute: "2-digit",
    });
  }

  // Auto-fill duration from selected exam when creating
  function onExamChange(examId: string) {
    const exam = exams.find(e => e._id === examId);
    setForm(f => ({
      ...f,
      examId,
      duration: exam ? String(exam.duration) : f.duration,
    }));
  }

  return (
    <div style={{ display: "flex", minHeight: "100vh" }} className="animated-bg">
      <Sidebar role="admin" />

      <main style={{ flex: 1, padding: "36px 40px", overflowY: "auto" }}>
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 28 }}>
          <div>
            <h1 style={{ color: "var(--text-primary)", fontSize: 26, fontWeight: 800, marginBottom: 4 }}>
              Assign Exams
            </h1>
            <p style={{ color: "var(--text-muted)", fontSize: 14 }}>
              Schedule examinations for students with start time, end time, and duration.
            </p>
          </div>
          <button
            id="assign-exam-btn"
            onClick={() => openAssign()}
            className="btn-glow"
            style={{ padding: "11px 22px", borderRadius: 12, fontSize: 14, fontWeight: 600 }}
          >
            + Assign Exam
          </button>
        </div>

        {/* Stats */}
        <div style={{ display: "flex", gap: 12, marginBottom: 24, flexWrap: "wrap" }}>
          {[
            { label: "Total", value: total, color: "#7c3aed" },
            { label: "Active", value: assignments.filter(a => a.computedStatus === "active").length, color: "#10b981" },
            { label: "Upcoming", value: assignments.filter(a => a.computedStatus === "upcoming" || a.computedStatus === "assigned").length, color: "#f59e0b" },
            { label: "Completed", value: assignments.filter(a => a.computedStatus === "completed").length, color: "#6b7280" },
          ].map(s => (
            <div key={s.label} style={{
              padding: "10px 20px", borderRadius: 10,
              background: "var(--bg-elevated)", border: "1px solid var(--bg-border)",
              fontSize: 13, color: "var(--text-secondary)",
            }}>
              <span style={{ fontWeight: 800, color: s.color, marginRight: 6, fontSize: 16 }}>{s.value}</span>
              {s.label}
            </div>
          ))}
        </div>

        {/* Filters */}
        <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap", alignItems: "center" }}>
          <input
            className="ez-input"
            placeholder="Search by name, student ID, or email…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ maxWidth: 320 }}
          />
          <select
            className="ez-input"
            value={filterExam}
            onChange={(e) => { setFilterExam(e.target.value); setPage(1); }}
            style={{ maxWidth: 200 }}
          >
            <option value="all">All Exams</option>
            {exams.map(ex => (
              <option key={ex._id} value={ex._id}>{ex.title}</option>
            ))}
          </select>
          <select
            className="ez-input"
            value={filterStatus}
            onChange={(e) => { setFilterStatus(e.target.value); setPage(1); }}
            style={{ maxWidth: 160 }}
          >
            <option value="all">All Statuses</option>
            <option value="assigned">Assigned</option>
            <option value="upcoming">Upcoming</option>
            <option value="started">In Progress</option>
            <option value="completed">Completed</option>
            <option value="expired">Expired</option>
            <option value="cancelled">Cancelled</option>
          </select>
        </div>

        {/* Table */}
        <div className="glass-card" style={{ overflow: "hidden" }}>
          <div style={{ overflowX: "auto" }}>
            <table className="ez-table">
              <thead>
                <tr>
                  <th>Student</th>
                  <th>Assigned Exam</th>
                  <th>Duration</th>
                  <th>Schedule</th>
                  <th>Status</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  [1, 2, 3, 4].map((i) => <SkeletonRow key={i} />)
                ) : assignments.length === 0 ? (
                  <tr>
                    <td colSpan={6} style={{ textAlign: "center", padding: "60px 0", color: "var(--text-muted)" }}>
                      <div style={{ fontSize: 40, marginBottom: 10 }}>📋</div>
                      <p>No assignments found</p>
                    </td>
                  </tr>
                ) : (
                  assignments.map((a) => (
                    <tr key={a._id}>
                      <td>
                        <div style={{ fontWeight: 600, color: "var(--text-primary)", fontSize: 13 }}>
                          {a.studentObjectId?.name || a.studentId}
                        </div>
                        <div style={{ color: "#a78bfa", fontSize: 11, fontFamily: "monospace", marginTop: 2 }}>
                          {a.studentObjectId?.studentId || a.studentId}
                        </div>
                        <div style={{ color: "var(--text-muted)", fontSize: 11 }}>
                          {a.studentObjectId?.email}
                        </div>
                      </td>
                      <td>
                        <div style={{ color: "var(--text-primary)", fontSize: 13, fontWeight: 600 }}>
                          {a.examId?.title || "—"}
                        </div>
                      </td>
                      <td>
                        <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>
                          {a.duration} min
                        </span>
                      </td>
                      <td>
                        <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>
                          <div>▶ {formatDateTime(a.startTime)}</div>
                          <div style={{ color: "var(--text-muted)", marginTop: 2 }}>⏹ {formatDateTime(a.endTime)}</div>
                        </div>
                      </td>
                      <td>
                        <StatusBadge status={a.computedStatus || a.status} />
                      </td>
                      <td>
                        <div style={{ display: "flex", gap: 8 }}>
                          {!["completed", "cancelled", "started"].includes(a.status) && (
                            <button
                              onClick={() => openAssign(a)}
                              style={{
                                padding: "5px 12px", borderRadius: 8, fontSize: 12, fontWeight: 600,
                                background: "rgba(124,58,237,0.08)", border: "1px solid rgba(124,58,237,0.25)",
                                color: "#a78bfa", cursor: "pointer", transition: "all 0.15s",
                              }}
                              onMouseOver={(e) => (e.currentTarget.style.background = "rgba(124,58,237,0.15)")}
                              onMouseOut={(e) => (e.currentTarget.style.background = "rgba(124,58,237,0.08)")}
                            >
                              Edit
                            </button>
                          )}
                          {!["completed", "cancelled"].includes(a.status) && (
                            <button
                              onClick={() => setCancelId(a._id)}
                              style={{
                                padding: "5px 12px", borderRadius: 8, fontSize: 12, fontWeight: 600,
                                background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.25)",
                                color: "#ef4444", cursor: "pointer", transition: "all 0.15s",
                              }}
                              onMouseOver={(e) => (e.currentTarget.style.background = "rgba(239,68,68,0.15)")}
                              onMouseOut={(e) => (e.currentTarget.style.background = "rgba(239,68,68,0.08)")}
                            >
                              Cancel
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div style={{
              display: "flex", justifyContent: "space-between", alignItems: "center",
              padding: "16px 24px", borderTop: "1px solid var(--bg-border)",
            }}>
              <span style={{ color: "var(--text-muted)", fontSize: 13 }}>
                Showing {assignments.length} of {total} assignments
              </span>
              <div style={{ display: "flex", gap: 8 }}>
                <button
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                  disabled={page === 1}
                  style={{
                    padding: "7px 14px", borderRadius: 8, fontSize: 13, fontWeight: 600,
                    background: "var(--bg-elevated)", border: "1px solid var(--bg-border)",
                    color: page === 1 ? "var(--text-muted)" : "var(--text-primary)",
                    cursor: page === 1 ? "not-allowed" : "pointer",
                  }}
                >
                  ← Prev
                </button>
                <span style={{
                  padding: "7px 14px", borderRadius: 8, fontSize: 13,
                  background: "rgba(124,58,237,0.1)", color: "#a78bfa", fontWeight: 700,
                }}>
                  {page} / {totalPages}
                </span>
                <button
                  onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                  disabled={page === totalPages}
                  style={{
                    padding: "7px 14px", borderRadius: 8, fontSize: 13, fontWeight: 600,
                    background: "var(--bg-elevated)", border: "1px solid var(--bg-border)",
                    color: page === totalPages ? "var(--text-muted)" : "var(--text-primary)",
                    cursor: page === totalPages ? "not-allowed" : "pointer",
                  }}
                >
                  Next →
                </button>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Assign / Edit Modal */}
      <Modal
        isOpen={modalOpen}
        onClose={() => { setModalOpen(false); setForm(emptyForm); setEditingId(null); }}
        title={editingId ? "Edit Assignment" : "Assign Exam to Student"}
        size="md"
        footer={
          <>
            <button
              onClick={() => { setModalOpen(false); setForm(emptyForm); setEditingId(null); }}
              style={{ padding: "9px 18px", borderRadius: 8, background: "var(--bg-elevated)", border: "1px solid var(--bg-border)", color: "var(--text-secondary)", cursor: "pointer" }}
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={saving}
              className="btn-glow"
              style={{ padding: "9px 20px", borderRadius: 8, fontSize: 14 }}
            >
              {saving ? "Saving…" : editingId ? "Update Assignment" : "Assign Exam"}
            </button>
          </>
        }
      >
        <form onSubmit={handleSave}>
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            {/* Student */}
            <div>
              <label style={labelStyle}>Student</label>
              <select
                className="ez-input"
                value={form.studentObjectId}
                onChange={(e) => setForm({ ...form, studentObjectId: e.target.value })}
                disabled={!!editingId}
              >
                <option value="">— Select Student —</option>
                {students.filter(s => s.isActive).map(s => (
                  <option key={s._id} value={s._id}>
                    {s.studentId} — {s.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Exam */}
            <div>
              <label style={labelStyle}>Exam</label>
              <select
                className="ez-input"
                value={form.examId}
                onChange={(e) => onExamChange(e.target.value)}
              >
                <option value="">— Select Exam —</option>
                {exams.map(ex => (
                  <option key={ex._id} value={ex._id}>
                    {ex.title} ({ex.duration} min)
                  </option>
                ))}
              </select>
            </div>

            {/* Date/Time grid */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div>
                <label style={labelStyle}>Start Date</label>
                <input type="date" className="ez-input" value={form.startDate}
                  onChange={(e) => setForm({ ...form, startDate: e.target.value })} />
              </div>
              <div>
                <label style={labelStyle}>Start Time</label>
                <input type="time" className="ez-input" value={form.startTime}
                  onChange={(e) => setForm({ ...form, startTime: e.target.value })} />
              </div>
            </div>

            {/* Duration */}
            <div>
              <label style={labelStyle}>Duration (minutes)</label>
              <input
                type="number"
                className="ez-input"
                min={1}
                placeholder="e.g. 60"
                value={form.duration}
                onChange={(e) => setForm({ ...form, duration: e.target.value })}
              />
            </div>

            {/* Notes */}
            <div>
              <label style={labelStyle}>Notes (optional)</label>
              <input
                className="ez-input"
                placeholder="Any special instructions…"
                value={form.notes}
                onChange={(e) => setForm({ ...form, notes: e.target.value })}
              />
            </div>
          </div>
        </form>
      </Modal>

      {/* Cancel Confirm */}
      <ConfirmDialog
        isOpen={!!cancelId}
        onClose={() => setCancelId(null)}
        onConfirm={handleCancel}
        title="Cancel Assignment"
        message="Cancel this exam assignment? The student will no longer have access to this exam."
        confirmLabel="Cancel Assignment"
        danger
      />
    </div>
  );
}

const labelStyle: React.CSSProperties = {
  color: "var(--text-secondary)",
  fontSize: 12,
  fontWeight: 600,
  letterSpacing: "0.05em",
  textTransform: "uppercase",
  display: "block",
  marginBottom: 8,
};
