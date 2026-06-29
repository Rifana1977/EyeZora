"use client";

import { useState, useEffect } from "react";
import Sidebar from "@/app/components/Sidebar";
import { adminApi } from "@/lib/api";
import { SkeletonQuestion } from "@/app/components/ui/Skeleton";
import Pagination from "@/app/components/ui/Pagination";
import Modal, { ConfirmDialog } from "@/app/components/ui/Modal";
import { toast } from "@/app/components/ui/Toast";

interface Question {
  _id: string;
  questionNumber: number;
  questionText: string;
  options: string[];
  correctOptionIndex: number;
  marks: number;
}

interface Exam {
  _id: string;
  title: string;
  duration: number;
  isActive?: boolean;
}

interface ExamData {
  exam: Exam;
  questions: Question[];
  pagination: { total: number; page: number; limit: number; totalPages: number };
}

interface QuestionDraft {
  questionText: string;
  options: string[];
  correctOptionIndex: number;
  marks: number;
}

const OPTION_LABELS = ["A", "B", "C", "D", "E"];

export default function QuestionsPage() {
  const [exams, setExams] = useState<Exam[]>([]);
  const [loadingExams, setLoadingExams] = useState(false);
  const [selectedExamId, setSelectedExamId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  const [data, setData] = useState<ExamData | null>(null);
  const [loadingQuestions, setLoadingQuestions] = useState(false);
  const [error, setError] = useState("");
  const [page, setPage] = useState(1);

  // Modals state
  const [createOpen, setCreateOpen] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [newDuration, setNewDuration] = useState(60);
  const [newQuestions, setNewQuestions] = useState<QuestionDraft[]>([
    { questionText: "", options: ["", "", "", ""], correctOptionIndex: 0, marks: 1 }
  ]);
  const [savingTest, setSavingTest] = useState(false);

  // Edit Test Modal
  const [editExam, setEditExam] = useState<Exam | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [editDuration, setEditDuration] = useState(60);
  const [savingExam, setSavingExam] = useState(false);

  // Delete Test Confirm
  const [deleteExamId, setDeleteExamId] = useState<string | null>(null);

  // Add Question Modal (single question)
  const [addQOpen, setAddQOpen] = useState(false);
  const [addQText, setAddQText] = useState("");
  const [addQOptions, setAddQOptions] = useState(["", "", "", ""]);
  const [addQCorrect, setAddQCorrect] = useState(0);
  const [addQMarks, setAddQMarks] = useState(1);
  const [savingQ, setSavingQ] = useState(false);

  // Edit Question Modal
  const [editQ, setEditQ] = useState<Question | null>(null);
  const [editLoading, setEditLoading] = useState(false);

  // Delete Question Confirm
  const [deleteQId, setDeleteQId] = useState<string | null>(null);

  // Load all exams on mount
  useEffect(() => {
    loadExams();
  }, []);

  const loadExams = async () => {
    setLoadingExams(true);
    try {
      const list = await adminApi.getAllExams();
      setExams(list);
    } catch (err: any) {
      toast.error(err.message || "Failed to load tests");
    } finally {
      setLoadingExams(false);
    }
  };

  const selectExam = async (id: string, p = 1) => {
    setSelectedExamId(id);
    setPage(p);
    setLoadingQuestions(true);
    setError("");
    try {
      const result = await adminApi.getTestQuestions(id, p, 20);
      setData(result);
    } catch (err: any) {
      setError(err.message || "Failed to load questions");
      setData(null);
    } finally {
      setLoadingQuestions(false);
    }
  };

  const handleCreateTest = async () => {
    if (!newTitle.trim()) {
      toast.error("Please enter a test title");
      return;
    }
    if (newDuration <= 0) {
      toast.error("Please enter a positive duration");
      return;
    }

    // Validate questions if any are entered
    for (let i = 0; i < newQuestions.length; i++) {
      const q = newQuestions[i];
      if (!q.questionText.trim()) {
        toast.error(`Question ${i + 1} text is empty`);
        return;
      }
      if (q.options.some((opt) => !opt.trim())) {
        toast.error(`Question ${i + 1} has one or more empty options`);
        return;
      }
      if (q.marks <= 0) {
        toast.error(`Question ${i + 1} must have a score greater than 0`);
        return;
      }
    }

    setSavingTest(true);
    try {
      // 1. Create Exam
      const exam = await adminApi.createExam(newTitle.trim(), newDuration);
      
      // 2. Add Questions sequentially
      for (const q of newQuestions) {
        await adminApi.addQuestion({
          examId: exam._id,
          questionText: q.questionText.trim(),
          options: q.options.map((o) => o.trim()),
          correctOptionIndex: q.correctOptionIndex,
          marks: q.marks,
        });
      }

      toast.success("Test and questions created successfully!");
      setCreateOpen(false);
      
      // Reset State
      setNewTitle("");
      setNewDuration(60);
      setNewQuestions([{ questionText: "", options: ["", "", "", ""], correctOptionIndex: 0, marks: 1 }]);

      // Reload list and select new test
      await loadExams();
      selectExam(exam._id, 1);
    } catch (err: any) {
      toast.error(err.message || "Failed to create test");
    } finally {
      setSavingTest(false);
    }
  };

  const handleEditTest = async () => {
    if (!editExam) return;
    if (!editTitle.trim()) {
      toast.error("Title cannot be empty");
      return;
    }
    if (editDuration <= 0) {
      toast.error("Duration must be a positive number");
      return;
    }

    setSavingExam(true);
    try {
      await adminApi.updateExam(editExam._id, editTitle.trim(), editDuration);
      toast.success("Test details updated!");
      setEditExam(null);
      await loadExams();
      if (selectedExamId === editExam._id) {
        selectExam(editExam._id, page);
      }
    } catch (err: any) {
      toast.error(err.message || "Failed to update test");
    } finally {
      setSavingExam(false);
    }
  };

  const handleDeleteTest = async () => {
    if (!deleteExamId) return;
    try {
      await adminApi.deleteExam(deleteExamId);
      toast.success("Test and its questions deleted successfully");
      setDeleteExamId(null);
      if (selectedExamId === deleteExamId) {
        setSelectedExamId(null);
        setData(null);
      }
      loadExams();
    } catch (err: any) {
      toast.error(err.message || "Failed to delete test");
    }
  };

  const handleAddQuestion = async () => {
    if (!selectedExamId) return;
    if (!addQText.trim()) {
      toast.error("Question text is required");
      return;
    }
    if (addQOptions.some((o) => !o.trim())) {
      toast.error("Please fill all options");
      return;
    }
    if (addQMarks <= 0) {
      toast.error("Marks must be greater than 0");
      return;
    }

    setSavingQ(true);
    try {
      await adminApi.addQuestion({
        examId: selectedExamId,
        questionText: addQText.trim(),
        options: addQOptions.map((o) => o.trim()),
        correctOptionIndex: addQCorrect,
        marks: addQMarks,
      });

      toast.success("Question added successfully!");
      setAddQOpen(false);

      // Reset fields
      setAddQText("");
      setAddQOptions(["", "", "", ""]);
      setAddQCorrect(0);
      setAddQMarks(1);

      // Reload selected test questions
      selectExam(selectedExamId, page);
    } catch (err: any) {
      toast.error(err.message || "Failed to add question");
    } finally {
      setSavingQ(false);
    }
  };

  const handleEditQuestion = async () => {
    if (!editQ || !selectedExamId) return;
    if (!editQ.questionText.trim()) {
      toast.error("Question text is required");
      return;
    }
    if (editQ.options.some((o) => !o.trim())) {
      toast.error("Please fill all options");
      return;
    }
    if (editQ.marks <= 0) {
      toast.error("Marks must be greater than 0");
      return;
    }

    setEditLoading(true);
    try {
      await adminApi.updateQuestion(editQ._id, {
        questionText: editQ.questionText.trim(),
        options: editQ.options.map((o) => o.trim()),
        correctOptionIndex: editQ.correctOptionIndex,
        marks: editQ.marks,
      });

      toast.success("Question updated!");
      setEditQ(null);
      selectExam(selectedExamId, page);
    } catch (err: any) {
      toast.error(err.message || "Failed to update question");
    } finally {
      setEditLoading(false);
    }
  };

  const handleDeleteQuestion = async () => {
    if (!deleteQId || !selectedExamId) return;
    try {
      await adminApi.deleteQuestion(deleteQId);
      toast.success("Question deleted");
      setDeleteQId(null);
      selectExam(selectedExamId, page);
    } catch (err: any) {
      toast.error(err.message || "Failed to delete question");
    }
  };

  const addQuestionDraft = () => {
    setNewQuestions([
      ...newQuestions,
      { questionText: "", options: ["", "", "", ""], correctOptionIndex: 0, marks: 1 }
    ]);
  };

  const removeQuestionDraft = (index: number) => {
    if (newQuestions.length <= 1) return;
    setNewQuestions(newQuestions.filter((_, idx) => idx !== index));
  };

  const handleDraftTextChange = (index: number, text: string) => {
    const updated = [...newQuestions];
    updated[index].questionText = text;
    setNewQuestions(updated);
  };

  const handleDraftOptionChange = (qIndex: number, optIndex: number, val: string) => {
    const updated = [...newQuestions];
    updated[qIndex].options[optIndex] = val;
    setNewQuestions(updated);
  };

  const handleDraftCorrectChange = (qIndex: number, correctIdx: number) => {
    const updated = [...newQuestions];
    updated[qIndex].correctOptionIndex = correctIdx;
    setNewQuestions(updated);
  };

  const handleDraftMarksChange = (qIndex: number, marks: number) => {
    const updated = [...newQuestions];
    updated[qIndex].marks = marks;
    setNewQuestions(updated);
  };

  const handleAddQOptionChange = (idx: number, val: string) => {
    const updated = [...addQOptions];
    updated[idx] = val;
    setAddQOptions(updated);
  };

  // Filter exams by query
  const filteredExams = exams.filter((ex) => {
    const query = searchQuery.toLowerCase().trim();
    return (
      ex.title.toLowerCase().includes(query) ||
      ex._id.toLowerCase().includes(query)
    );
  });

  return (
    <div style={{ display: "flex", minHeight: "100vh" }} className="animated-bg">
      <Sidebar role="admin" />

      <main style={{ flex: 1, padding: "36px 40px", overflowY: "auto", display: "flex", flexDirection: "column" }}>
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 28 }}>
          <div>
            <h1 style={{ color: "var(--text-primary)", fontSize: 26, fontWeight: 800, marginBottom: 4 }}>
              Test & Question Management
            </h1>
            <p style={{ color: "var(--text-muted)", fontSize: 14 }}>
              Manage tests, create exams, and maintain question lifecycle
            </p>
          </div>
          <button
            onClick={() => setCreateOpen(true)}
            className="btn-glow"
            style={{ padding: "11px 22px", borderRadius: 12, fontSize: 14, fontWeight: 600 }}
          >
            + Create New Test
          </button>
        </div>

        {/* Dynamic Split Layout */}
        <div style={{ display: "flex", gap: 28, flex: 1, minHeight: 0 }}>
          
          {/* Left Panel: Available Tests */}
          <div style={{ width: 340, display: "flex", flexDirection: "column", gap: 16 }}>
            <div className="glass-card" style={{ padding: 20, display: "flex", flexDirection: "column", gap: 14 }}>
              <h3 style={{ color: "var(--text-primary)", fontSize: 16, fontWeight: 700, margin: 0 }}>
                Available Exams ({exams.length})
              </h3>
              <input
                className="ez-input"
                placeholder="Filter by title or ObjectId…"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{ fontSize: 13, padding: "10px 14px" }}
              />
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 12, overflowY: "auto", maxHeight: "60vh" }}>
              {loadingExams ? (
                <div style={{ textAlign: "center", color: "var(--text-muted)", padding: 20 }}>
                  Loading exams list…
                </div>
              ) : filteredExams.length === 0 ? (
                <div style={{ textAlign: "center", color: "var(--text-muted)", padding: 24, background: "rgba(255,255,255,0.02)", border: "1px dashed var(--bg-border)", borderRadius: 12 }}>
                  No exams found
                </div>
              ) : (
                filteredExams.map((ex) => {
                  const selected = ex._id === selectedExamId;
                  return (
                    <div
                      key={ex._id}
                      onClick={() => selectExam(ex._id, 1)}
                      style={{
                        padding: "16px 20px",
                        borderRadius: 14,
                        background: selected
                          ? "linear-gradient(135deg, rgba(124, 58, 237, 0.2), rgba(91, 33, 182, 0.1))"
                          : "var(--bg-elevated)",
                        border: selected
                          ? "1px solid rgba(124, 58, 237, 0.5)"
                          : "1px solid var(--bg-border)",
                        cursor: "pointer",
                        display: "flex",
                        flexDirection: "column",
                        gap: 10,
                        transition: "all 0.2s",
                      }}
                      onMouseOver={(e) => { if (!selected) e.currentTarget.style.borderColor = "rgba(124,58,237,0.3)"; }}
                      onMouseOut={(e) => { if (!selected) e.currentTarget.style.borderColor = "var(--bg-border)"; }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8 }}>
                        <h4 style={{ color: selected ? "#a78bfa" : "var(--text-primary)", fontSize: 15, fontWeight: 700, margin: 0, wordBreak: "break-word" }}>
                          {ex.title}
                        </h4>
                        <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setEditExam(ex);
                              setEditTitle(ex.title);
                              setEditDuration(ex.duration);
                            }}
                            style={{ background: "none", border: "none", color: "var(--text-muted)", cursor: "pointer", fontSize: 13 }}
                            title="Edit Exam"
                          >
                            ✏️
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setDeleteExamId(ex._id);
                            }}
                            style={{ background: "none", border: "none", color: "#ef4444", cursor: "pointer", fontSize: 13 }}
                            title="Delete Exam"
                          >
                            🗑️
                          </button>
                        </div>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--text-muted)" }}>
                        <span>⏱ {ex.duration} mins</span>
                        <span style={{ fontFamily: "monospace" }}>ID: {ex._id.slice(-6)}…</span>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {/* Right Panel: Selected Test Questions */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
            {data ? (
              <div style={{ display: "flex", flexDirection: "column", gap: 20, flex: 1 }}>
                
                {/* Selected Exam Banner */}
                <div className="glass-card" style={{
                  padding: "20px 24px",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  background: "linear-gradient(135deg, rgba(124, 58, 237, 0.08), rgba(91, 33, 182, 0.04))",
                }}>
                  <div>
                    <h2 style={{ color: "var(--text-primary)", fontSize: 19, fontWeight: 800, margin: "0 0 4px 0" }}>
                      {data.exam.title}
                    </h2>
                    <span style={{ color: "var(--text-muted)", fontSize: 12 }}>
                      Duration: {data.exam.duration} mins · Test ObjectId: <span style={{ fontFamily: "monospace", color: "#a78bfa" }}>{data.exam._id}</span>
                    </span>
                  </div>
                  
                  <button
                    onClick={() => setAddQOpen(true)}
                    className="btn-glow"
                    style={{ padding: "8px 18px", borderRadius: 10, fontSize: 13, fontWeight: 700 }}
                  >
                    + Add Question
                  </button>
                </div>

                {error && (
                  <div style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 12, padding: "16px 20px", color: "#ef4444" }}>
                    ❌ {error}
                  </div>
                )}

                {/* Questions Loader */}
                {loadingQuestions ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                    {[1, 2].map((i) => <SkeletonQuestion key={i} />)}
                  </div>
                ) : data.questions.length === 0 ? (
                  <div style={{ textAlign: "center", padding: "60px 0", color: "var(--text-muted)", background: "rgba(255,255,255,0.01)", border: "1px dashed var(--bg-border)", borderRadius: 16 }}>
                    <div style={{ fontSize: 32, marginBottom: 12 }}>❓</div>
                    <p style={{ fontWeight: 600 }}>No questions in this test yet</p>
                    <p style={{ fontSize: 12, marginTop: 4 }}>Click "+ Add Question" to register questions for this test.</p>
                  </div>
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                    {data.questions.map((q) => (
                      <QuestionCard
                        key={q._id}
                        q={q}
                        onEdit={() => setEditQ({ ...q })}
                        onDelete={() => setDeleteQId(q._id)}
                      />
                    ))}

                    <Pagination
                      currentPage={data.pagination.page}
                      totalPages={data.pagination.totalPages}
                      onPageChange={(p) => selectExam(data.exam._id, p)}
                    />
                  </div>
                )}
              </div>
            ) : (
              <div className="glass-card" style={{ display: "flex", flex: 1, flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 48, textAlign: "center", color: "var(--text-muted)" }}>
                <div style={{ fontSize: 64, marginBottom: 16, opacity: 0.3 }}>📋</div>
                <h3 style={{ color: "var(--text-primary)", fontSize: 18, fontWeight: 700, marginBottom: 8 }}>
                  No Test Selected
                </h3>
                <p style={{ fontSize: 13, maxWidth: 360, lineHeight: 1.5 }}>
                  Select an exam from the left panel to review questions, or click "+ Create New Test" to define a new exam with multiple questions.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* CREATE TEST MODAL (supports adding multiple questions) */}
      <Modal
        isOpen={createOpen}
        onClose={() => { if (!savingTest) setCreateOpen(false); }}
        title="Create New Test"
        size="lg"
        footer={
          <>
            <button
              onClick={() => setCreateOpen(false)}
              disabled={savingTest}
              style={{
                padding: "9px 18px", borderRadius: 8,
                background: "rgba(255,255,255,0.06)",
                border: "1px solid var(--bg-border)",
                color: "var(--text-secondary)", cursor: savingTest ? "not-allowed" : "pointer",
              }}
            >
              Cancel
            </button>
            <button
              onClick={handleCreateTest}
              disabled={savingTest}
              className="btn-glow"
              style={{ padding: "9px 24px", borderRadius: 8, fontSize: 14, opacity: savingTest ? 0.7 : 1 }}
            >
              {savingTest ? "Creating Test & Questions…" : "Save Test & Questions"}
            </button>
          </>
        }
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 20, maxHeight: "65vh", overflowY: "auto", paddingRight: 8 }}>
          
          {/* Test Meta Info */}
          <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16 }}>
            <div>
              <label style={{ color: "var(--text-secondary)", fontSize: 12, fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase", display: "block", marginBottom: 6 }}>
                Test Title
              </label>
              <input
                className="ez-input"
                placeholder="e.g. Midterm Physics Exam"
                value={newTitle}
                onChange={(e) => setNewTitle(e.target.value)}
              />
            </div>
            <div>
              <label style={{ color: "var(--text-secondary)", fontSize: 12, fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase", display: "block", marginBottom: 6 }}>
                Duration (minutes)
              </label>
              <input
                className="ez-input"
                type="number"
                min={1}
                value={newDuration}
                onChange={(e) => setNewDuration(Number(e.target.value))}
              />
            </div>
          </div>

          <hr style={{ border: "none", borderTop: "1px solid var(--bg-border)", margin: "4px 0" }} />

          {/* Dynamic Questions Builder */}
          <div>
            <div style={{ display: "flex", justifySelf: "stretch", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <h3 style={{ color: "var(--text-primary)", fontSize: 16, fontWeight: 700, margin: 0 }}>
                Questions ({newQuestions.length})
              </h3>
              <button
                type="button"
                onClick={addQuestionDraft}
                style={{
                  background: "rgba(124,58,237,0.12)", border: "1px solid rgba(124,58,237,0.3)",
                  color: "#a78bfa", padding: "6px 14px", borderRadius: 8, fontSize: 12, fontWeight: 700, cursor: "pointer"
                }}
              >
                ➕ Add Another Question
              </button>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              {newQuestions.map((q, idx) => (
                <div key={idx} style={{
                  padding: 20,
                  borderRadius: 14,
                  background: "rgba(255,255,255,0.02)",
                  border: "1px solid var(--bg-border)",
                  position: "relative",
                }}>
                  {/* Question Header */}
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                    <span style={{
                      padding: "3px 10px", borderRadius: 6, background: "rgba(124, 58, 237, 0.2)",
                      color: "#a78bfa", fontSize: 12, fontWeight: 800
                    }}>
                      Question #{idx + 1}
                    </span>
                    {newQuestions.length > 1 && (
                      <button
                        type="button"
                        onClick={() => removeQuestionDraft(idx)}
                        style={{ background: "none", border: "none", color: "#ef4444", fontSize: 12, cursor: "pointer", fontWeight: 600 }}
                      >
                        ✕ Remove
                      </button>
                    )}
                  </div>

                  {/* Question text */}
                  <div style={{ marginBottom: 12 }}>
                    <input
                      className="ez-input"
                      placeholder={`Enter question text #${idx + 1}`}
                      value={q.questionText}
                      onChange={(e) => handleDraftTextChange(idx, e.target.value)}
                    />
                  </div>

                  {/* Options */}
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 12 }}>
                    {q.options.map((opt, oIdx) => (
                      <div key={oIdx} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <input
                          type="radio"
                          name={`correct-option-${idx}`}
                          checked={q.correctOptionIndex === oIdx}
                          onChange={() => handleDraftCorrectChange(idx, oIdx)}
                          style={{ accentColor: "#7c3aed", width: 16, height: 16, cursor: "pointer" }}
                        />
                        <span style={{ fontSize: 12, color: "var(--text-muted)", fontWeight: 700 }}>
                          {OPTION_LABELS[oIdx]}
                        </span>
                        <input
                          className="ez-input"
                          placeholder={`Option ${oIdx + 1}`}
                          value={opt}
                          onChange={(e) => handleDraftOptionChange(idx, oIdx, e.target.value)}
                          style={{ padding: "8px 12px", fontSize: 13 }}
                        />
                      </div>
                    ))}
                  </div>

                  {/* Marks */}
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <label style={{ color: "var(--text-muted)", fontSize: 12, fontWeight: 600 }}>Marks:</label>
                    <input
                      className="ez-input"
                      type="number"
                      min={1}
                      value={q.marks}
                      onChange={(e) => handleDraftMarksChange(idx, Number(e.target.value))}
                      style={{ width: 80, padding: "6px 10px", fontSize: 13 }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Modal>

      {/* EDIT TEST BANNER DETAILS MODAL */}
      {editExam && (
        <Modal
          isOpen={!!editExam}
          onClose={() => setEditExam(null)}
          title="Edit Test Details"
          size="sm"
          footer={
            <>
              <button
                onClick={() => setEditExam(null)}
                style={{ padding: "8px 16px", borderRadius: 8, background: "rgba(255,255,255,0.06)", border: "1px solid var(--bg-border)", color: "var(--text-secondary)", cursor: "pointer" }}
              >
                Cancel
              </button>
              <button
                onClick={handleEditTest}
                disabled={savingExam}
                className="btn-glow"
                style={{ padding: "8px 18px", borderRadius: 8 }}
              >
                {savingExam ? "Saving…" : "Save Changes"}
              </button>
            </>
          }
        >
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div>
              <label style={{ color: "var(--text-secondary)", fontSize: 12, display: "block", marginBottom: 6 }}>Test Title</label>
              <input
                className="ez-input"
                value={editTitle}
                onChange={(e) => setEditTitle(e.target.value)}
              />
            </div>
            <div>
              <label style={{ color: "var(--text-secondary)", fontSize: 12, display: "block", marginBottom: 6 }}>Duration (minutes)</label>
              <input
                className="ez-input"
                type="number"
                min={1}
                value={editDuration}
                onChange={(e) => setEditDuration(Number(e.target.value))}
              />
            </div>
          </div>
        </Modal>
      )}

      {/* DELETE TEST CONFIRM DIALOG */}
      <ConfirmDialog
        isOpen={!!deleteExamId}
        onClose={() => setDeleteExamId(null)}
        onConfirm={handleDeleteTest}
        title="Delete Test"
        message="Are you sure you want to delete this test? All questions registered under this test will be permanently deleted."
        confirmLabel="Delete Test"
        danger
      />

      {/* ADD SINGLE QUESTION MODAL */}
      <Modal
        isOpen={addQOpen}
        onClose={() => setAddQOpen(false)}
        title="Add Single Question"
        size="md"
        footer={
          <>
            <button
              onClick={() => setAddQOpen(false)}
              style={{ padding: "8px 16px", borderRadius: 8, background: "rgba(255,255,255,0.06)", border: "1px solid var(--bg-border)", color: "var(--text-secondary)", cursor: "pointer" }}
            >
              Cancel
            </button>
            <button
              onClick={handleAddQuestion}
              disabled={savingQ}
              className="btn-glow"
              style={{ padding: "8px 18px", borderRadius: 8 }}
            >
              {savingQ ? "Saving Question…" : "Add Question"}
            </button>
          </>
        }
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div>
            <label style={{ color: "var(--text-secondary)", fontSize: 12, display: "block", marginBottom: 6 }}>Question Text</label>
            <input
              className="ez-input"
              placeholder="Enter question text"
              value={addQText}
              onChange={(e) => setAddQText(e.target.value)}
            />
          </div>

          <div>
            <label style={{ color: "var(--text-secondary)", fontSize: 12, display: "block", marginBottom: 6 }}>Options</label>
            {addQOptions.map((opt, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                <input
                  type="radio"
                  name="single-q-correct"
                  checked={addQCorrect === i}
                  onChange={() => setAddQCorrect(i)}
                  style={{ accentColor: "#7c3aed", width: 16, height: 16 }}
                />
                <span style={{ fontSize: 12, color: "var(--text-muted)", fontWeight: 700 }}>
                  {OPTION_LABELS[i]}
                </span>
                <input
                  className="ez-input"
                  placeholder={`Option ${i + 1}`}
                  value={opt}
                  onChange={(e) => handleAddQOptionChange(i, e.target.value)}
                  style={{ padding: "8px 12px", fontSize: 13 }}
                />
              </div>
            ))}
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <label style={{ color: "var(--text-secondary)", fontSize: 12 }}>Marks:</label>
            <input
              className="ez-input"
              type="number"
              min={1}
              value={addQMarks}
              onChange={(e) => setAddQMarks(Number(e.target.value))}
              style={{ width: 80, padding: "8px 12px", fontSize: 13 }}
            />
          </div>
        </div>
      </Modal>

      {/* EDIT QUESTION MODAL */}
      {editQ && (
        <Modal
          isOpen={!!editQ}
          onClose={() => setEditQ(null)}
          title={`Edit Question #${editQ.questionNumber}`}
          size="lg"
          footer={
            <>
              <button
                onClick={() => setEditQ(null)}
                style={{
                  padding: "9px 18px", borderRadius: 8,
                  background: "rgba(255,255,255,0.06)",
                  border: "1px solid var(--bg-border)",
                  color: "var(--text-secondary)", cursor: "pointer",
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleEditQuestion}
                disabled={editLoading}
                className="btn-glow"
                style={{ padding: "9px 20px", borderRadius: 8, fontSize: 14 }}
              >
                {editLoading ? "Saving…" : "Save Changes"}
              </button>
            </>
          }
        >
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div>
              <label style={{ color: "var(--text-secondary)", fontSize: 12, fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase", display: "block", marginBottom: 8 }}>
                Question Text
              </label>
              <textarea
                className="ez-input"
                rows={3}
                value={editQ.questionText}
                onChange={(e) => setEditQ({ ...editQ, questionText: e.target.value })}
                style={{ resize: "vertical" }}
              />
            </div>

            <div>
              <label style={{ color: "var(--text-secondary)", fontSize: 12, fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase", display: "block", marginBottom: 8 }}>
                Options (select correct answer)
              </label>
              {editQ.options.map((opt, idx) => (
                <div key={idx} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                  <input
                    type="radio"
                    name="edit-q-correct"
                    checked={editQ.correctOptionIndex === idx}
                    onChange={() => setEditQ({ ...editQ, correctOptionIndex: idx })}
                    style={{ width: 18, height: 18, accentColor: "#7c3aed" }}
                  />
                  <span style={{
                    width: 28, height: 28, borderRadius: 6,
                    background: editQ.correctOptionIndex === idx ? "rgba(124,58,237,0.3)" : "rgba(255,255,255,0.06)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 12, fontWeight: 700, color: editQ.correctOptionIndex === idx ? "#a78bfa" : "var(--text-muted)",
                    flexShrink: 0,
                  }}>
                    {OPTION_LABELS[idx]}
                  </span>
                  <input
                    className="ez-input"
                    value={opt}
                    onChange={(e) => {
                      const opts = [...editQ.options];
                      opts[idx] = e.target.value;
                      setEditQ({ ...editQ, options: opts });
                    }}
                    style={{ flex: 1 }}
                  />
                </div>
              ))}
            </div>

            <div style={{ display: "flex", gap: 16 }}>
              <div style={{ flex: 1 }}>
                <label style={{ color: "var(--text-secondary)", fontSize: 12, fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase", display: "block", marginBottom: 8 }}>
                  Marks
                </label>
                <input
                  className="ez-input"
                  type="number"
                  min={1}
                  value={editQ.marks}
                  onChange={(e) => setEditQ({ ...editQ, marks: Number(e.target.value) })}
                />
              </div>
            </div>
          </div>
        </Modal>
      )}

      {/* DELETE QUESTION CONFIRM DIALOG */}
      <ConfirmDialog
        isOpen={!!deleteQId}
        onClose={() => setDeleteQId(null)}
        onConfirm={handleDeleteQuestion}
        title="Delete Question"
        message="Are you sure you want to delete this question? This action cannot be undone and will renumber remaining questions."
        confirmLabel="Delete"
        danger
      />
    </div>
  );
}

// ── Question Card ─────────────────────────────────────────────────────────────

function QuestionCard({
  q,
  onEdit,
  onDelete,
}: {
  q: Question;
  onEdit: () => void;
  onDelete: () => void;
}) {
  return (
    <div className="glass-card fade-in" style={{ padding: 24, marginBottom: 16, position: "relative" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{
            width: 32, height: 32, borderRadius: 8,
            background: "rgba(124,58,237,0.2)",
            display: "flex", alignItems: "center", justifyContent: "center",
            color: "#a78bfa", fontSize: 14, fontWeight: 800,
          }}>
            {q.questionNumber}
          </span>
          <span style={{ color: "var(--text-muted)", fontSize: 12 }}>
            {q.marks} mark{q.marks !== 1 ? "s" : ""}
          </span>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <ActionBtn label="Edit" icon="✎" onClick={onEdit} color="#7c3aed" />
          <ActionBtn label="Delete" icon="✕" onClick={onDelete} color="#ef4444" />
        </div>
      </div>

      <p style={{ color: "var(--text-primary)", fontSize: 15, fontWeight: 600, marginBottom: 16, lineHeight: 1.5 }}>
        {q.questionText}
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        {q.options.map((opt, idx) => {
          const isCorrect = idx === q.correctOptionIndex;
          return (
            <div key={idx} style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              padding: "10px 14px",
              borderRadius: 10,
              background: isCorrect ? "rgba(16,185,129,0.1)" : "rgba(255,255,255,0.04)",
              border: isCorrect
                ? "1px solid rgba(16,185,129,0.4)"
                : "1px solid rgba(255,255,255,0.06)",
            }}>
              <span style={{
                width: 24, height: 24, borderRadius: 6,
                background: isCorrect ? "rgba(16,185,129,0.25)" : "rgba(255,255,255,0.06)",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 11, fontWeight: 700,
                color: isCorrect ? "#10b981" : "var(--text-muted)",
                flexShrink: 0,
              }}>
                {OPTION_LABELS[idx]}
              </span>
              <span style={{ color: isCorrect ? "#10b981" : "var(--text-secondary)", fontSize: 13 }}>
                {opt}
              </span>
              {isCorrect && (
                <span style={{ marginLeft: "auto", color: "#10b981", fontSize: 16 }}>✓</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ActionBtn({ label, icon, onClick, color }: { label: string; icon: string; onClick: () => void; color: string }) {
  return (
    <button
      onClick={onClick}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        padding: "7px 14px",
        borderRadius: 8,
        background: `${color}12`,
        border: `1px solid ${color}30`,
        color,
        fontSize: 12,
        fontWeight: 600,
        cursor: "pointer",
        transition: "all 0.15s",
      }}
      onMouseOver={(e) => (e.currentTarget.style.background = `${color}22`)}
      onMouseOut={(e) => (e.currentTarget.style.background = `${color}12`)}
    >
      <span>{icon}</span> {label}
    </button>
  );
}
