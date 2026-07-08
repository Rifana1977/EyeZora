"use client";

import { useEffect, useState, useCallback } from "react";
import Sidebar from "@/app/components/Sidebar";
import { adminApi } from "@/lib/api";
import Modal, { ConfirmDialog } from "@/app/components/ui/Modal";
import { toast } from "@/app/components/ui/Toast";
import { SkeletonRow } from "@/app/components/ui/Skeleton";
import { useRowSelection } from "@/lib/hooks/useRowSelection";
import { SelectAllCheckbox } from "@/app/components/ui/SelectAllCheckbox";
import { BulkActionToolbar } from "@/app/components/ui/BulkActionToolbar";

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

  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Bulk assignment states
  const [bulkModalOpen, setBulkModalOpen] = useState(false);
  const [bulkForm, setBulkForm] = useState({
    examId: "",
    startDate: "",
    startTime: "",
    duration: "",
    notes: "",
  });
  const [bulkSelectedIds, setBulkSelectedIds] = useState<string[]>([]);
  const [bulkStudentSearch, setBulkStudentSearch] = useState("");
  const [bulkSaving, setBulkSaving] = useState(false);
  const [bulkResults, setBulkResults] = useState<any>(null);

  // Bulk row selection (for table delete)
  const sel = useRowSelection();
  const [bulkDeleteOpen, setBulkDeleteOpen] = useState(false);
  const [bulkDeleting, setBulkDeleting] = useState(false);

  function openBulkAssign() {
    setBulkForm({
      examId: "",
      startDate: "",
      startTime: "",
      duration: "",
      notes: "",
    });
    setBulkSelectedIds([]);
    setBulkStudentSearch("");
    setBulkResults(null);
    setBulkModalOpen(true);
  }

  function onBulkExamChange(examId: string) {
    const exam = exams.find(e => e._id === examId);
    setBulkForm(f => ({
      ...f,
      examId,
      duration: exam ? String(exam.duration) : f.duration,
    }));
  }

  async function handleBulkSave(e: React.FormEvent) {
    e.preventDefault();
    const { examId, startDate, startTime, duration } = bulkForm;

    if (!examId || !startDate || !startTime || !duration) {
      toast.error("Exam, Start Date, Start Time, and Duration are required");
      return;
    }

    if (bulkSelectedIds.length === 0) {
      toast.error("Please select at least one student");
      return;
    }

    const startDateTime = new Date(`${startDate}T${startTime}:00`);
    if (isNaN(startDateTime.getTime())) {
      toast.error("Invalid start date/time");
      return;
    }

    setBulkSaving(true);
    try {
      const res = await adminApi.bulkAssign({
        studentObjectIds: bulkSelectedIds,
        examId,
        startTime: startDateTime.toISOString(),
        duration: Number(duration),
        notes: bulkForm.notes,
      });
      setBulkResults(res);
      toast.success(`Successfully assigned exam to ${res.assigned} students!`);
      load();
    } catch (err: any) {
      toast.error(err.message || "Bulk assignment failed");
    } finally {
      setBulkSaving(false);
    }
  }

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

  async function handleDelete() {
    if (!deleteId) return;
    setDeleting(true);
    try {
      await adminApi.deleteAssignment(deleteId);
      toast.success("Assignment deleted successfully.");
      setDeleteId(null);
      sel.clearSelection();
      load();
    } catch (err: any) {
      toast.error(err.message || "Failed to delete assignment.");
    } finally {
      setDeleting(false);
    }
  }

  async function handleBulkDelete() {
    if (sel.selectedCount === 0) return;
    setBulkDeleting(true);
    try {
      const res = await adminApi.bulkDeleteAssignments(sel.selectedArray);
      toast.success(res.message || `${sel.selectedCount} assignment(s) deleted successfully.`);
      sel.clearSelection();
      setBulkDeleteOpen(false);
      load();
    } catch (err: any) {
      toast.error(err.message || "Bulk delete failed.");
    } finally {
      setBulkDeleting(false);
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
          <div style={{ display: "flex", gap: 12 }}>
            <button
              id="bulk-assign-btn"
              onClick={() => openBulkAssign()}
              className="btn-glow"
              style={{
                padding: "11px 22px", borderRadius: 12, fontSize: 14, fontWeight: 600,
                background: "rgba(124,58,237,0.1)", color: "#a78bfa",
                border: "1px solid rgba(124,58,237,0.3)", cursor: "pointer"
              }}
            >
              📥 Bulk Assign
            </button>
            <button
              id="assign-exam-btn"
              onClick={() => openAssign()}
              className="btn-glow"
              style={{ padding: "11px 22px", borderRadius: 12, fontSize: 14, fontWeight: 600 }}
            >
              + Assign Exam
            </button>
          </div>
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
            onChange={(e) => { setSearch(e.target.value); sel.clearSelection(); }}
            style={{ maxWidth: 320 }}
          />
          <select
            className="ez-input"
            value={filterExam}
            onChange={(e) => { setFilterExam(e.target.value); setPage(1); sel.clearSelection(); }}
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
            onChange={(e) => { setFilterStatus(e.target.value); setPage(1); sel.clearSelection(); }}
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

        {/* Bulk Action Toolbar */}
        <BulkActionToolbar
          count={sel.selectedCount}
          onDelete={() => setBulkDeleteOpen(true)}
          onClear={sel.clearSelection}
          deleting={bulkDeleting}
          deleteLabel="🗑 Delete Selected"
          noun="assignments"
        />

        {/* Table */}
        <div className="glass-card" style={{ overflow: "hidden" }}>
          <div style={{ overflowX: "auto" }}>
            <table className="ez-table">
              <thead>
                <tr>
                  <th style={{ width: 40, textAlign: "center" }}>
                    <SelectAllCheckbox
                      checked={sel.isAllSelected(assignments.map(a => a._id))}
                      indeterminate={sel.isIndeterminate(assignments.map(a => a._id))}
                      onChange={() => sel.toggleAll(assignments.map(a => a._id))}
                      disabled={loading || assignments.length === 0}
                    />
                  </th>
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
                    <td colSpan={7} style={{ textAlign: "center", padding: "60px 0", color: "var(--text-muted)" }}>
                      <div style={{ fontSize: 40, marginBottom: 10 }}>📋</div>
                      <p>No assignments found</p>
                    </td>
                  </tr>
                ) : (
                  assignments.map((a) => {
                    const isSelected = sel.selected.has(a._id);
                    return (
                      <tr
                        key={a._id}
                        style={{
                          background: isSelected ? "rgba(124,58,237,0.06)" : undefined,
                          transition: "background 0.15s",
                        }}
                      >
                        <td style={{ textAlign: "center" }}>
                          <input
                            type="checkbox"
                            checked={isSelected}
                            onChange={() => sel.toggleOne(a._id)}
                            style={{ accentColor: "#7c3aed", width: 15, height: 15, cursor: "pointer" }}
                          />
                        </td>
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
                          <button
                            onClick={() => setDeleteId(a._id)}
                            style={{
                              padding: "5px 12px", borderRadius: 8, fontSize: 12, fontWeight: 600,
                              background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.25)",
                              color: "#ef4444", cursor: "pointer", transition: "all 0.15s",
                            }}
                            onMouseOver={(e) => (e.currentTarget.style.background = "rgba(239,68,68,0.15)")}
                            onMouseOut={(e) => (e.currentTarget.style.background = "rgba(239,68,68,0.08)")}
                          >
                            Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                  })
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
                  onClick={() => { setPage(p => Math.max(1, p - 1)); sel.clearSelection(); }}
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
                  onClick={() => { setPage(p => Math.min(totalPages, p + 1)); sel.clearSelection(); }}
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

      {/* Bulk Assign Modal */}
      <Modal
        isOpen={bulkModalOpen}
        onClose={() => { setBulkModalOpen(false); }}
        title="Bulk Assign Exam"
        size="lg"
        footer={
          <>
            <button
              onClick={() => setBulkModalOpen(false)}
              style={{ padding: "9px 18px", borderRadius: 8, background: "var(--bg-elevated)", border: "1px solid var(--bg-border)", color: "var(--text-secondary)", cursor: "pointer" }}
            >
              {bulkResults ? "Close" : "Cancel"}
            </button>
            {!bulkResults && (
              <button
                onClick={handleBulkSave}
                disabled={bulkSaving || bulkSelectedIds.length === 0}
                className="btn-glow"
                style={{ padding: "9px 20px", borderRadius: 8, fontSize: 14 }}
              >
                {bulkSaving ? "Assigning…" : `Assign to ${bulkSelectedIds.length} Students`}
              </button>
            )}
          </>
        }
      >
        {!bulkResults ? (
          <form onSubmit={handleBulkSave}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
              {/* Left Column — Config */}
              <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                <h4 style={{ color: "var(--text-accent)", fontSize: 13, fontWeight: 700, textTransform: "uppercase", marginBottom: 4 }}>
                  1. Schedule details
                </h4>
                
                <div>
                  <label style={labelStyle}>Exam</label>
                  <select
                    className="ez-input"
                    value={bulkForm.examId}
                    onChange={(e) => onBulkExamChange(e.target.value)}
                  >
                    <option value="">— Select Exam —</option>
                    {exams.map(ex => (
                      <option key={ex._id} value={ex._id}>{ex.title} ({ex.duration} min)</option>
                    ))}
                  </select>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  <div>
                    <label style={labelStyle}>Start Date</label>
                    <input type="date" className="ez-input" value={bulkForm.startDate}
                      onChange={(e) => setBulkForm({ ...bulkForm, startDate: e.target.value })} />
                  </div>
                  <div>
                    <label style={labelStyle}>Start Time</label>
                    <input type="time" className="ez-input" value={bulkForm.startTime}
                      onChange={(e) => setBulkForm({ ...bulkForm, startTime: e.target.value })} />
                  </div>
                </div>

                <div>
                  <label style={labelStyle}>Duration (minutes)</label>
                  <input
                    type="number"
                    className="ez-input"
                    min={1}
                    value={bulkForm.duration}
                    onChange={(e) => setBulkForm({ ...bulkForm, duration: e.target.value })}
                  />
                </div>

                <div>
                  <label style={labelStyle}>Notes (optional)</label>
                  <input
                    className="ez-input"
                    placeholder="Instructions…"
                    value={bulkForm.notes}
                    onChange={(e) => setBulkForm({ ...bulkForm, notes: e.target.value })}
                  />
                </div>
              </div>

              {/* Right Column — Student Selection */}
              <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                <h4 style={{ color: "var(--text-accent)", fontSize: 13, fontWeight: 700, textTransform: "uppercase", marginBottom: 4 }}>
                  2. Select Students ({bulkSelectedIds.length} selected)
                </h4>
                
                <input
                  className="ez-input"
                  placeholder="Filter student list…"
                  value={bulkStudentSearch}
                  onChange={(e) => setBulkStudentSearch(e.target.value)}
                />

                <div style={{
                  border: "1px solid var(--bg-border)", borderRadius: 10,
                  maxHeight: 220, overflowY: "auto", background: "var(--bg-elevated)",
                  padding: 8
                }}>
                  {/* Select All Toggle */}
                  <div style={{
                    display: "flex", alignItems: "center", gap: 10, padding: "8px 10px",
                    borderBottom: "1px solid var(--bg-border)", marginBottom: 8
                  }}>
                    <input
                      type="checkbox"
                      id="select-all-students"
                      checked={students.length > 0 && bulkSelectedIds.length === students.filter(s => s.isActive).length}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setBulkSelectedIds(students.filter(s => s.isActive).map(s => s._id));
                        } else {
                          setBulkSelectedIds([]);
                        }
                      }}
                      style={{ accentColor: "#7c3aed" }}
                    />
                    <label htmlFor="select-all-students" style={{ fontWeight: 700, color: "var(--text-primary)", fontSize: 12, cursor: "pointer" }}>
                      Select All Active Students
                    </label>
                  </div>

                  {students
                    .filter(s => s.isActive && (
                      s.name.toLowerCase().includes(bulkStudentSearch.toLowerCase()) ||
                      s.studentId.toLowerCase().includes(bulkStudentSearch.toLowerCase())
                    ))
                    .map(s => {
                      const checked = bulkSelectedIds.includes(s._id);
                      return (
                        <div key={s._id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "6px 10px" }}>
                          <input
                            type="checkbox"
                            id={`bulk-stu-${s._id}`}
                            checked={checked}
                            onChange={() => {
                              if (checked) {
                                setBulkSelectedIds(bulkSelectedIds.filter(id => id !== s._id));
                              } else {
                                setBulkSelectedIds([...bulkSelectedIds, s._id]);
                              }
                            }}
                            style={{ accentColor: "#7c3aed" }}
                          />
                          <label htmlFor={`bulk-stu-${s._id}`} style={{ color: "var(--text-secondary)", fontSize: 12, cursor: "pointer" }}>
                            <strong>{s.studentId}</strong> — {s.name}
                          </label>
                        </div>
                      );
                    })}
                </div>
              </div>
            </div>
          </form>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{
              padding: 16, borderRadius: 10, background: "rgba(16,185,129,0.08)",
              border: "1px solid rgba(16,185,129,0.2)", display: "flex", alignItems: "center", gap: 12
            }}>
              <span style={{ fontSize: 24 }}>✅</span>
              <div>
                <h4 style={{ color: "#10b981", fontSize: 14, fontWeight: 700, margin: 0 }}>Bulk Assign Finished</h4>
                <p style={{ color: "var(--text-secondary)", fontSize: 12, margin: "2px 0 0" }}>
                  Assigned successfully to <strong>{bulkResults.assigned}</strong> students. Failed/Conflicts: <strong>{bulkResults.failed}</strong>.
                </p>
              </div>
            </div>

            {bulkResults.errors && bulkResults.errors.length > 0 && (
              <div>
                <h5 style={{ color: "#ef4444", fontSize: 12, fontWeight: 700, marginBottom: 6 }}>Scheduling Conflicts & Errors ({bulkResults.errors.length})</h5>
                <div style={{
                  maxHeight: 200, overflowY: "auto", border: "1px solid var(--bg-border)",
                  borderRadius: 8, padding: 8, background: "var(--bg-base)"
                }}>
                  {bulkResults.errors.map((err: any, idx: number) => (
                    <div key={idx} style={{
                      fontSize: 11, color: "var(--text-secondary)", paddingBottom: 6,
                      borderBottom: idx < bulkResults.errors.length - 1 ? "1px solid var(--bg-border)" : "none",
                      marginBottom: 4
                    }}>
                      <strong>{err.studentName || err.studentId || "Student"}:</strong> <span style={{ color: "#ef4444" }}>{err.error}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </Modal>

      {/* Delete Confirm (single) */}
      <ConfirmDialog
        isOpen={!!deleteId}
        onClose={() => setDeleteId(null)}
        onConfirm={handleDelete}
        title="Delete Assignment"
        message="Are you sure you want to permanently remove this assignment record? This action cannot be undone."
        confirmLabel="Delete"
        danger
        loading={deleting}
      />

      {/* Bulk Delete Confirm */}
      <ConfirmDialog
        isOpen={bulkDeleteOpen}
        onClose={() => setBulkDeleteOpen(false)}
        onConfirm={handleBulkDelete}
        title="Delete Selected Assignments"
        message={`Are you sure you want to delete ${sel.selectedCount} assignment${sel.selectedCount !== 1 ? "s" : ""}? This will permanently remove the assignment records. This action cannot be undone.`}
        confirmLabel={`Delete ${sel.selectedCount} Assignment${sel.selectedCount !== 1 ? "s" : ""}`}
        danger
        loading={bulkDeleting}
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
