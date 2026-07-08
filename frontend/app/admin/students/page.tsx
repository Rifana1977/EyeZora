"use client";

import { useEffect, useState } from "react";
import Sidebar from "@/app/components/Sidebar";
import { adminApi } from "@/lib/api";
import Modal, { ConfirmDialog } from "@/app/components/ui/Modal";
import { toast } from "@/app/components/ui/Toast";
import { SkeletonRow } from "@/app/components/ui/Skeleton";
import { useRowSelection } from "@/lib/hooks/useRowSelection";
import { SelectAllCheckbox } from "@/app/components/ui/SelectAllCheckbox";
import { BulkActionToolbar } from "@/app/components/ui/BulkActionToolbar";

interface Student {
  _id: string;
  studentId: string;
  name: string;
  email: string;
  isActive: boolean;
  createdAt: string;
}

const emptyForm = { studentId: "", name: "", email: "", password: "", isActive: true };

export default function StudentsPage() {
  const [students, setStudents] = useState<Student[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");

  const [addOpen, setAddOpen] = useState(false);
  const [form, setForm] = useState(emptyForm);
  const [saving, setSaving] = useState(false);

  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [deleteName, setDeleteName] = useState("");

  // Bulk selection
  const sel = useRowSelection();
  const [bulkDeleteOpen, setBulkDeleteOpen] = useState(false);
  const [bulkDeleting, setBulkDeleting] = useState(false);

  // Bulk import states
  const [importOpen, setImportOpen] = useState(false);
  const [importFile, setImportFile] = useState<File | null>(null);
  const [sendEmails, setSendEmails] = useState(false);
  const [importing, setImporting] = useState(false);
  const [importResults, setImportResults] = useState<any>(null);

  function downloadCredentialsCSV(credentials: any[]) {
    const headers = ["Student ID", "Name", "Email", "Temporary Password"];
    const rows = credentials.map(c => [c.studentId, c.name, c.email, c.tempPassword]);
    const csvContent = "data:text/csv;charset=utf-8,\uFEFF"
      + [headers.join(","), ...rows.map(e => e.map(val => `"${val}"`).join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `eyezora_credentials_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  async function handleImport(e: React.FormEvent) {
    e.preventDefault();
    if (!importFile) {
      toast.error("Please select a file to import");
      return;
    }
    setImporting(true);
    try {
      const res = await adminApi.bulkImportStudents(importFile, sendEmails);
      setImportResults(res);
      toast.success(`Successfully imported ${res.created} students!`);
      if (res.credentials && res.credentials.length > 0) {
        downloadCredentialsCSV(res.credentials);
      }
      load();
    } catch (err: any) {
      toast.error(err.message || "Failed to import students");
    } finally {
      setImporting(false);
    }
  }

  async function load() {
    setLoading(true);
    try {
      const s = await adminApi.getStudents();
      setStudents(s);
    } catch (err: any) {
      toast.error(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  async function handleAdd(e: React.FormEvent) {
    e.preventDefault();
    if (!form.studentId || !form.name || !form.email || !form.password) {
      toast.error("Student ID, name, email, and password are required");
      return;
    }
    setSaving(true);
    try {
      await adminApi.registerStudent({
        studentId: form.studentId,
        name: form.name,
        email: form.email,
        password: form.password,
        isActive: form.isActive,
      });
      toast.success(`Student ${form.name} registered successfully!`);
      setForm(emptyForm);
      setAddOpen(false);
      load();
    } catch (err: any) {
      toast.error(err.message);
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete() {
    if (!deleteId) return;
    try {
      await adminApi.deleteStudent(deleteId);
      toast.success("Student removed");
      setDeleteId(null);
      sel.clearSelection();
      load();
    } catch (err: any) {
      toast.error(err.message);
    }
  }

  async function handleBulkDelete() {
    setBulkDeleting(true);
    try {
      const res = await adminApi.bulkDeleteStudents(sel.selectedArray);
      const { deleted, requested } = res;
      if (deleted === requested) {
        toast.success(`${deleted} student${deleted !== 1 ? "s" : ""} deleted successfully.`);
      } else {
        toast.success(`Deleted ${deleted} of ${requested} students.`);
      }
      sel.clearSelection();
      setBulkDeleteOpen(false);
      load();
    } catch (err: any) {
      toast.error(err.message || "Bulk delete failed");
    } finally {
      setBulkDeleting(false);
    }
  }

  const filtered = students.filter((s) =>
    s.name.toLowerCase().includes(search.toLowerCase()) ||
    s.studentId.toLowerCase().includes(search.toLowerCase()) ||
    s.email.toLowerCase().includes(search.toLowerCase())
  );

  const filteredIds = filtered.map((s) => s._id);

  return (
    <div style={{ display: "flex", minHeight: "100vh" }} className="animated-bg">
      <Sidebar role="admin" />

      <main style={{ flex: 1, padding: "36px 40px", overflowY: "auto" }}>
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 28 }}>
          <div>
            <h1 style={{ color: "var(--text-primary)", fontSize: 26, fontWeight: 800, marginBottom: 4 }}>
              Student Management
            </h1>
            <p style={{ color: "var(--text-muted)", fontSize: 14 }}>
              Register student accounts. Use <strong style={{ color: "var(--text-accent)" }}>Assign Exams</strong> to schedule examinations.
            </p>
          </div>
          <div style={{ display: "flex", gap: 12 }}>
            <button
              id="bulk-import-btn"
              onClick={() => { setImportResults(null); setImportFile(null); setImportOpen(true); }}
              className="btn-glow"
              style={{
                padding: "11px 22px", borderRadius: 12, fontSize: 14, fontWeight: 600,
                background: "rgba(124,58,237,0.1)", color: "#a78bfa",
                border: "1px solid rgba(124,58,237,0.3)", cursor: "pointer"
              }}
            >
              📥 Bulk Import
            </button>
            <button
              id="add-student-btn"
              onClick={() => setAddOpen(true)}
              className="btn-glow"
              style={{ padding: "11px 22px", borderRadius: 12, fontSize: 14, fontWeight: 600 }}
            >
              + Register Student
            </button>
          </div>
        </div>

        {/* Search */}
        <div style={{ marginBottom: 20 }}>
          <input
            className="ez-input"
            placeholder="Search by name, ID, or email…"
            value={search}
            onChange={(e) => { setSearch(e.target.value); sel.clearSelection(); }}
            style={{ maxWidth: 400 }}
          />
        </div>

        {/* Stats */}
        <div style={{ display: "flex", gap: 12, marginBottom: 20 }}>
          <div style={{
            padding: "10px 18px", borderRadius: 10,
            background: "var(--bg-elevated)", border: "1px solid var(--bg-border)",
            fontSize: 13, color: "var(--text-secondary)",
          }}>
            <span style={{ fontWeight: 700, color: "var(--text-primary)", marginRight: 6 }}>
              {students.length}
            </span>
            Total Students
          </div>
          <div style={{
            padding: "10px 18px", borderRadius: 10,
            background: "rgba(16,185,129,0.08)", border: "1px solid rgba(16,185,129,0.2)",
            fontSize: 13, color: "#10b981",
          }}>
            <span style={{ fontWeight: 700, marginRight: 6 }}>
              {students.filter(s => s.isActive).length}
            </span>
            Active
          </div>
        </div>

        {/* Bulk Action Toolbar */}
        <BulkActionToolbar
          count={sel.selectedCount}
          onDelete={() => setBulkDeleteOpen(true)}
          onClear={sel.clearSelection}
          deleting={bulkDeleting}
          noun="students"
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
                  <th>Student ID</th>
                  <th>Name</th>
                  <th>Email</th>
                  <th>Status</th>
                  <th>Registered</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  [1, 2, 3].map((i) => <SkeletonRow key={i} />)
                ) : filtered.length === 0 ? (
                  <tr>
                    <td colSpan={7} style={{ textAlign: "center", padding: "60px 0", color: "var(--text-muted)" }}>
                      {search ? "No students match your search" : "No students registered yet"}
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
                          <span style={{ fontFamily: "monospace", color: "#a78bfa", fontWeight: 700, fontSize: 13 }}>
                            {s.studentId}
                          </span>
                        </td>
                        <td style={{ color: "var(--text-primary)", fontWeight: 600 }}>{s.name}</td>
                        <td style={{ color: "var(--text-secondary)", fontSize: 13 }}>{s.email}</td>
                        <td>
                          <span style={{
                            padding: "3px 10px",
                            borderRadius: 999,
                            fontSize: 11,
                            fontWeight: 700,
                            background: s.isActive ? "rgba(16,185,129,0.12)" : "rgba(100,116,139,0.12)",
                            color: s.isActive ? "#10b981" : "#64748b",
                            border: `1px solid ${s.isActive ? "rgba(16,185,129,0.3)" : "rgba(100,116,139,0.3)"}`,
                          }}>
                            {s.isActive ? "Active" : "Inactive"}
                          </span>
                        </td>
                        <td style={{ fontSize: 12, color: "var(--text-muted)" }}>
                          {new Date(s.createdAt).toLocaleDateString("en-IN")}
                        </td>
                        <td>
                          <button
                            onClick={async () => {
                              try {
                                const updatedStatus = !s.isActive;
                                await adminApi.updateStudent(s._id, { isActive: updatedStatus });
                                toast.success(`Student ${s.name} is now ${updatedStatus ? "Active" : "Inactive"}`);
                                load();
                              } catch (err: any) {
                                toast.error(err.message || "Failed to update status");
                              }
                            }}
                            style={{
                              padding: "6px 14px",
                              borderRadius: 8,
                              background: s.isActive ? "rgba(239,68,68,0.08)" : "rgba(16,185,129,0.08)",
                              border: `1px solid ${s.isActive ? "rgba(239,68,68,0.2)" : "rgba(16,185,129,0.2)"}`,
                              color: s.isActive ? "#ef4444" : "#10b981",
                              fontSize: 12,
                              fontWeight: 600,
                              cursor: "pointer",
                              marginRight: 8,
                              transition: "all 0.15s",
                            }}
                            onMouseOver={(e) => (e.currentTarget.style.background = s.isActive ? "rgba(239,68,68,0.15)" : "rgba(16,185,129,0.15)")}
                            onMouseOut={(e) => (e.currentTarget.style.background = s.isActive ? "rgba(239,68,68,0.08)" : "rgba(16,185,129,0.08)")}
                          >
                            {s.isActive ? "Deactivate" : "Activate"}
                          </button>
                          <button
                            onClick={() => { setDeleteId(s._id); setDeleteName(s.name); }}
                            style={{
                              padding: "6px 14px",
                              borderRadius: 8,
                              background: "rgba(239,68,68,0.08)",
                              border: "1px solid rgba(239,68,68,0.25)",
                              color: "#ef4444",
                              fontSize: 12,
                              fontWeight: 600,
                              cursor: "pointer",
                              transition: "all 0.15s",
                            }}
                            onMouseOver={(e) => (e.currentTarget.style.background = "rgba(239,68,68,0.15)")}
                            onMouseOut={(e) => (e.currentTarget.style.background = "rgba(239,68,68,0.08)")}
                          >
                            Remove
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

      {/* Add Student Modal */}
      <Modal
        isOpen={addOpen}
        onClose={() => { setAddOpen(false); setForm(emptyForm); }}
        title="Register New Student"
        size="md"
        footer={
          <>
            <button
              onClick={() => { setAddOpen(false); setForm(emptyForm); }}
              style={{ padding: "9px 18px", borderRadius: 8, background: "var(--bg-elevated)", border: "1px solid var(--bg-border)", color: "var(--text-secondary)", cursor: "pointer" }}
            >
              Cancel
            </button>
            <button
              onClick={handleAdd}
              disabled={saving}
              className="btn-glow"
              style={{ padding: "9px 20px", borderRadius: 8, fontSize: 14 }}
            >
              {saving ? "Registering…" : "Register Student"}
            </button>
          </>
        }
      >
        <form onSubmit={handleAdd}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <FormField label="Student ID" value={form.studentId} onChange={(v) => setForm({ ...form, studentId: v })} placeholder="e.g. STU001" />
            <FormField label="Full Name" value={form.name} onChange={(v) => setForm({ ...form, name: v })} placeholder="John Doe" />
            <FormField label="Email" type="email" value={form.email} onChange={(v) => setForm({ ...form, email: v })} placeholder="john@email.com" />
            <FormField label="Temp Password" type="password" value={form.password} onChange={(v) => setForm({ ...form, password: v })} placeholder="Set temporary password" />
          </div>

          {/* Status toggle */}
          <div style={{ marginTop: 16 }}>
            <label style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={form.isActive}
                onChange={(e) => setForm({ ...form, isActive: e.target.checked })}
                style={{ width: 16, height: 16, accentColor: "#7c3aed" }}
              />
              <span style={{ color: "var(--text-secondary)", fontSize: 13, fontWeight: 500 }}>
                Account Active (student can login immediately)
              </span>
            </label>
          </div>

          <div style={{
            marginTop: 16, padding: "12px 14px", borderRadius: 10,
            background: "rgba(124,58,237,0.06)", border: "1px solid rgba(124,58,237,0.15)",
          }}>
            <p style={{ color: "var(--text-muted)", fontSize: 12, margin: 0, lineHeight: 1.6 }}>
              💡 Exam assignment is managed separately. After registering, go to{" "}
              <strong style={{ color: "var(--text-accent)" }}>Assign Exams</strong> to schedule this student&apos;s examination.
            </p>
          </div>
        </form>
      </Modal>

      {/* Bulk Import Modal */}
      <Modal
        isOpen={importOpen}
        onClose={() => { setImportOpen(false); setImportFile(null); setImportResults(null); }}
        title="Bulk Import Students (CSV / Excel)"
        size="md"
        footer={
          <>
            <button
              onClick={() => { setImportOpen(false); setImportFile(null); setImportResults(null); }}
              style={{ padding: "9px 18px", borderRadius: 8, background: "var(--bg-elevated)", border: "1px solid var(--bg-border)", color: "var(--text-secondary)", cursor: "pointer" }}
            >
              Close
            </button>
            {!importResults && (
              <button
                onClick={handleImport}
                disabled={importing || !importFile}
                className="btn-glow"
                style={{ padding: "9px 20px", borderRadius: 8, fontSize: 14 }}
              >
                {importing ? "Importing…" : "Upload & Import"}
              </button>
            )}
          </>
        }
      >
        {!importResults ? (
          <form onSubmit={handleImport}>
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <p style={{ color: "var(--text-secondary)", fontSize: 13, lineHeight: 1.6 }}>
                Upload a CSV or Excel (.xlsx, .xls) file. The file should contain column headers:
                <strong style={{ color: "var(--text-accent)" }}> studentId, name, email</strong>.
              </p>

              <div style={{
                padding: "24px 14px", border: "2px dashed var(--bg-border)",
                borderRadius: 12, textAlign: "center", background: "var(--bg-elevated)",
                position: "relative", cursor: "pointer"
              }}>
                <input
                  type="file"
                  accept=".csv, application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, application/vnd.ms-excel"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) setImportFile(f);
                  }}
                  style={{
                    position: "absolute", top: 0, left: 0, width: "100%", height: "100%",
                    opacity: 0, cursor: "pointer"
                  }}
                />
                <div style={{ fontSize: 32, marginBottom: 8 }}>📄</div>
                <div style={{ color: "var(--text-primary)", fontSize: 14, fontWeight: 600 }}>
                  {importFile ? importFile.name : "Click to select CSV or Excel file"}
                </div>
                <div style={{ color: "var(--text-muted)", fontSize: 12, marginTop: 4 }}>
                  Max size: 10MB
                </div>
              </div>

              <div style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer" }}>
                <input
                  type="checkbox"
                  id="send-emails-check"
                  checked={sendEmails}
                  onChange={(e) => setSendEmails(e.target.checked)}
                  style={{ width: 16, height: 16, accentColor: "#7c3aed" }}
                />
                <label htmlFor="send-emails-check" style={{ color: "var(--text-secondary)", fontSize: 13, fontWeight: 500, cursor: "pointer" }}>
                  Email login credentials automatically to imported students
                </label>
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
                <h4 style={{ color: "#10b981", fontSize: 14, fontWeight: 700, margin: 0 }}>Import Completed Successfully</h4>
                <p style={{ color: "var(--text-secondary)", fontSize: 12, margin: "2px 0 0" }}>
                  Created <strong>{importResults.created}</strong> student accounts. Failed: <strong>{importResults.failed}</strong>.
                </p>
              </div>
            </div>

            {importResults.credentials && importResults.credentials.length > 0 && (
              <p style={{ color: "var(--text-muted)", fontSize: 12, margin: 0 }}>
                📥 Credentials CSV has been downloaded automatically.
              </p>
            )}

            {importResults.errors && importResults.errors.length > 0 && (
              <div>
                <h5 style={{ color: "#ef4444", fontSize: 12, fontWeight: 700, marginBottom: 6 }}>Import Errors ({importResults.errors.length})</h5>
                <div style={{
                  maxHeight: 150, overflowY: "auto", border: "1px solid var(--bg-border)",
                  borderRadius: 8, padding: 8, background: "var(--bg-base)"
                }}>
                  {importResults.errors.map((err: any, idx: number) => (
                    <div key={idx} style={{
                      fontSize: 11, color: "var(--text-secondary)", paddingBottom: 6,
                      borderBottom: idx < importResults.errors.length - 1 ? "1px solid var(--bg-border)" : "none",
                      marginBottom: 4
                    }}>
                      <strong>Row/ID:</strong> {err.studentId || err.row || "Unknown"} — <span style={{ color: "#ef4444" }}>{err.error}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </Modal>

      {/* Single Delete Confirm */}
      <ConfirmDialog
        isOpen={!!deleteId}
        onClose={() => { setDeleteId(null); setDeleteName(""); }}
        onConfirm={handleDelete}
        title="Remove Student"
        message={`Remove ${deleteName || "this student"}? They will no longer be able to login.`}
        confirmLabel="Remove"
        danger
      />

      {/* Bulk Delete Confirm */}
      <ConfirmDialog
        isOpen={bulkDeleteOpen}
        onClose={() => setBulkDeleteOpen(false)}
        onConfirm={handleBulkDelete}
        title="Delete Selected Students"
        message={`Are you sure you want to delete ${sel.selectedCount} student${sel.selectedCount !== 1 ? "s" : ""}? This action cannot be undone.`}
        confirmLabel={bulkDeleting ? "Deleting…" : `Delete ${sel.selectedCount} Student${sel.selectedCount !== 1 ? "s" : ""}`}
        danger
      />
    </div>
  );
}

function FormField({ label, value, onChange, placeholder, type = "text" }: {
  label: string; value: string; onChange: (v: string) => void;
  placeholder?: string; type?: string;
}) {
  return (
    <div>
      <label style={{ color: "var(--text-secondary)", fontSize: 12, fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase", display: "block", marginBottom: 8 }}>
        {label}
      </label>
      <input
        className="ez-input"
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
      />
    </div>
  );
}
