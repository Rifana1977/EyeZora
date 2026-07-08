/**
 * api.ts — Typed API wrapper with automatic auth header injection.
 * All API calls go through these helpers.
 */

import { getToken } from "./auth";

export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";
export const AI_BASE = process.env.NEXT_PUBLIC_AI_URL || "http://localhost:8000";

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(
  url: string,
  options: RequestInit = {}
): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(url, { ...options, headers });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: "Request failed" }));
    throw new ApiError(body.error || `HTTP ${res.status}`, res.status);
  }

  return res.json() as Promise<T>;
}

// ── Auth ───────────────────────────────────────────────────────────────────────

export const authApi = {
  adminLogin: (email: string, password: string) =>
    request<{ token: string; user: any }>(`${API_BASE}/api/auth/admin/login`, {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),

  studentLogin: (identifier: string, password: string) =>
    request<{ token: string; user: any }>(`${API_BASE}/api/auth/student/login`, {
      method: "POST",
      body: JSON.stringify({ identifier, password }),
    }),

  forgotPassword: (studentId: string, email: string) =>
    request<{ message: string }>(`${API_BASE}/api/auth/forgot-password`, {
      method: "POST",
      body: JSON.stringify({ studentId, email }),
    }),

  resetPassword: (token: string, password: string) =>
    request<{ message: string }>(`${API_BASE}/api/auth/reset-password`, {
      method: "POST",
      body: JSON.stringify({ token, password }),
    }),

  changePassword: (currentPassword: string, newPassword: string) =>
    request<{ message: string }>(`${API_BASE}/api/auth/student/change-password`, {
      method: "POST",
      body: JSON.stringify({ currentPassword, newPassword }),
    }),
};

// ── Admin ──────────────────────────────────────────────────────────────────────

export const adminApi = {
  getStats: () => request<any>(`${API_BASE}/api/admin/stats`),

  // Exams
  getAllExams: () => request<any[]>(`${API_BASE}/api/admin/exams`),
  createExam: (title: string, duration: number) =>
    request<any>(`${API_BASE}/api/admin/exam`, {
      method: "POST",
      body: JSON.stringify({ title, duration }),
    }),
  updateExam: (id: string, title: string, duration: number) =>
    request<any>(`${API_BASE}/api/admin/exam/${id}`, {
      method: "PUT",
      body: JSON.stringify({ title, duration }),
    }),
  deleteExam: (id: string) =>
    request<any>(`${API_BASE}/api/admin/exam/${id}`, { method: "DELETE" }),

  // Questions
  getTestQuestions: (testId: string, page = 1, limit = 20) =>
    request<any>(`${API_BASE}/api/admin/test/${testId}/questions?page=${page}&limit=${limit}`),
  updateQuestion: (id: string, data: any) =>
    request<any>(`${API_BASE}/api/admin/question/${id}`, {
      method: "PUT",
      body: JSON.stringify(data),
    }),
  deleteQuestion: (id: string) =>
    request<any>(`${API_BASE}/api/admin/question/${id}`, { method: "DELETE" }),
  addQuestion: (data: any) =>
    request<any>(`${API_BASE}/api/admin/question`, {
      method: "POST",
      body: JSON.stringify(data),
    }),

  // Students
  getStudents: () => request<any[]>(`${API_BASE}/api/admin/students`),
  registerStudent: (data: any) =>
    request<any>(`${API_BASE}/api/admin/students`, {
      method: "POST",
      body: JSON.stringify(data),
    }),
  updateStudent: (id: string, data: any) =>
    request<any>(`${API_BASE}/api/admin/students/${id}`, {
      method: "PUT",
      body: JSON.stringify(data),
    }),
  deleteStudent: (id: string) =>
    request<any>(`${API_BASE}/api/admin/students/${id}`, { method: "DELETE" }),

  // Bulk delete
  bulkDeleteStudents: (ids: string[]) =>
    request<{ deleted: number; requested: number }>(`${API_BASE}/api/admin/students/bulk-delete`, {
      method: "DELETE",
      body: JSON.stringify({ ids }),
    }),

  bulkCancelAssignments: (ids: string[]) =>
    request<{ cancelled: number; skipped: number; requested: number }>(
      `${API_BASE}/api/admin/assignments/bulk-cancel`,
      { method: "DELETE", body: JSON.stringify({ ids }) }
    ),

  bulkDeleteSessions: (ids: string[]) =>
    request<{ deleted: number; requested: number }>(`${API_BASE}/api/admin/sessions/bulk-delete`, {
      method: "DELETE",
      body: JSON.stringify({ ids }),
    }),

  bulkDeleteSubmissions: (ids: string[]) =>
    request<{ deleted: number; requested: number }>(`${API_BASE}/api/admin/results/bulk-delete`, {
      method: "DELETE",
      body: JSON.stringify({ ids }),
    }),

  // Assignments
  getAssignments: (params?: {
    search?: string;
    examId?: string;
    status?: string;
    page?: number;
    limit?: number;
  }) => {
    const qs = new URLSearchParams();
    if (params?.search) qs.set("search", params.search);
    if (params?.examId) qs.set("examId", params.examId);
    if (params?.status) qs.set("status", params.status);
    if (params?.page) qs.set("page", String(params.page));
    if (params?.limit) qs.set("limit", String(params.limit));
    return request<any>(`${API_BASE}/api/admin/assignments?${qs}`);
  },
  createAssignment: (data: any) =>
    request<any>(`${API_BASE}/api/admin/assignments`, {
      method: "POST",
      body: JSON.stringify(data),
    }),
  updateAssignment: (id: string, data: any) =>
    request<any>(`${API_BASE}/api/admin/assignments/${id}`, {
      method: "PUT",
      body: JSON.stringify(data),
    }),
  deleteAssignment: (id: string) =>
    request<any>(`${API_BASE}/api/admin/assignments/${id}`, { method: "DELETE" }),
  bulkDeleteAssignments: (assignmentIds: string[]) =>
    request<any>(`${API_BASE}/api/admin/assignments`, {
      method: "DELETE",
      body: JSON.stringify({ assignmentIds }),
    }),
  getStudentAssignment: (studentId: string) =>
    request<any>(`${API_BASE}/api/admin/assignments/student/${studentId}`),
  bulkAssign: (data: { studentObjectIds: string[]; examId: string; startTime: string; duration: number; notes?: string }) =>
    request<any>(`${API_BASE}/api/admin/assignments/bulk`, {
      method: "POST",
      body: JSON.stringify(data),
    }),

  // Monitoring
  getSessions: () => request<any[]>(`${API_BASE}/api/admin/sessions`),
  getSessionReport: (id: string) =>
    request<any>(`${API_BASE}/api/admin/sessions/${id}/report`),

  // Results
  getResults: (examId?: string) => {
    const qs = examId ? `?examId=${examId}` : "";
    return request<any[]>(`${API_BASE}/api/admin/results${qs}`);
  },
  publishResults: (data: { examId?: string; submissionIds?: string[] }) =>
    request<any>(`${API_BASE}/api/admin/results/publish`, {
      method: "POST",
      body: JSON.stringify(data),
    }),
  unpublishResults: (data: { examId?: string; submissionIds?: string[] }) =>
    request<any>(`${API_BASE}/api/admin/results/unpublish`, {
      method: "POST",
      body: JSON.stringify(data),
    }),

  // Bulk import
  bulkImportStudents: async (file: File, sendEmails: boolean): Promise<any> => {
    const token = getToken();
    const formData = new FormData();
    formData.append("file", file);
    formData.append("sendEmails", String(sendEmails));
    const res = await fetch(`${API_BASE}/api/admin/students/bulk-import`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token || ""}` },
      body: formData,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ error: "Import failed" }));
      throw new Error(body.error || "Import failed");
    }
    return res.json();
  },
};

// ── Student ────────────────────────────────────────────────────────────────────

export const studentApi = {
  getExamQuestions: (examId: string) =>
    request<any[]>(`${API_BASE}/api/student/exam/${examId}`),

  getMyResults: () =>
    request<any[]>(`${API_BASE}/api/student/results`),
};

// ── Session ────────────────────────────────────────────────────────────────────

export const sessionApi = {
  start: (examId: string, examTitle: string, assignmentId?: string) =>
    request<{ sessionId: string; resumed: boolean }>(`${API_BASE}/api/session/start`, {
      method: "POST",
      body: JSON.stringify({ examId, examTitle, assignmentId }),
    }),

  logEvent: (sessionId: string, event: string, confidence = 100) =>
    request<any>(`${API_BASE}/api/session/log`, {
      method: "POST",
      body: JSON.stringify({ sessionId, event, confidence }),
    }),

  end: (sessionId: string, answers: number[], examId: string) =>
    request<any>(`${API_BASE}/api/session/end`, {
      method: "POST",
      body: JSON.stringify({ sessionId, answers, examId }),
    }),

  /**
   * Upload video and/or audio recordings together in one request.
   */
  uploadRecording: async (
    sessionId: string,
    videoBlob: Blob | null,
    audioBlob: Blob | null
  ): Promise<{ videoUrl: string | null; audioUrl: string | null }> => {
    const token = getToken();
    const formData = new FormData();
    formData.append("sessionId", sessionId);
    if (videoBlob) {
      formData.append("video", videoBlob, "recording.webm");
    }
    if (audioBlob) {
      formData.append("audio", audioBlob, "audio.webm");
    }
    const res = await fetch(`${API_BASE}/api/session/recording`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token || ""}` },
      body: formData,
    });
    if (!res.ok) throw new Error("Recording upload failed");
    return res.json();
  },
};

// ── AI Service ─────────────────────────────────────────────────────────────────

export const aiApi = {
  analyzeFrame: async (
    imageBase64: string,
    studentId: string,
    examId: string,
    sessionId?: string
  ) => {
    try {
      const token = getToken();
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };
      if (token) {
        headers["Authorization"] = `Bearer ${token}`;
      }

      const res = await fetch(`${API_BASE}/api/session/analyze`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          image: imageBase64,
          student_id: studentId,
          exam_id: examId,
          session_id: sessionId,
        }),
      });
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP ${res.status}`);
      }
      return res.json();
    } catch (err: any) {
      throw err;
    }
  },

  /** Check if the Python AI service is reachable */
  healthCheck: async (): Promise<{ available: boolean; modelsLoaded: boolean }> => {
    try {
      const token = getToken();
      const headers: Record<string, string> = {};
      if (token) {
        headers["Authorization"] = `Bearer ${token}`;
      }

      const res = await fetch(`${API_BASE}/api/session/ai-health`, {
        headers,
        signal: AbortSignal.timeout(3000)
      });
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP ${res.status}`);
      }
      const data = await res.json();
      return { available: true, modelsLoaded: data.modelsLoaded === true };
    } catch (err: any) {
      console.error("AI service health check failed:", err.message);
      return { available: false, modelsLoaded: false };
    }
  },
};

export { ApiError };
