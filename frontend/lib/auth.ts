/**
 * auth.ts — JWT token helpers for the frontend.
 * Stores token in localStorage with a ez_ prefix for clarity.
 */

const TOKEN_KEY = "ez_token";
const USER_KEY = "ez_user";
const THEME_KEY = "ez_theme";

export interface AssignedExam {
  assignmentId: string;
  id: string;
  title: string;
  duration: number;       // minutes
  startTime: string;      // ISO date string
  endTime: string;        // ISO date string
  status: "upcoming" | "active" | "expired" | "started" | "completed" | "cancelled";
}

export interface AuthUser {
  id: string;
  name: string;
  email: string;
  role: "admin" | "student";
  studentId?: string;
  assignedExam?: AssignedExam | null;
}

export function saveAuth(token: string, user: AuthUser) {
  if (typeof window === "undefined") return;
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function getUser(): AuthUser | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as AuthUser;
  } catch {
    return null;
  }
}

export function clearAuth() {
  if (typeof window === "undefined") return;
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

export function isAuthenticated(): boolean {
  return !!getToken();
}

export function getRole(): "admin" | "student" | null {
  return getUser()?.role ?? null;
}

// ── Theme helpers ──────────────────────────────────────────────────────────────

export type Theme = "dark" | "light";

export function getTheme(): Theme {
  if (typeof window === "undefined") return "dark";
  return (localStorage.getItem(THEME_KEY) as Theme) || "dark";
}

export function setTheme(theme: Theme) {
  if (typeof window === "undefined") return;
  localStorage.setItem(THEME_KEY, theme);
  document.documentElement.setAttribute("data-theme", theme);
}

export function initTheme() {
  const theme = getTheme();
  document.documentElement.setAttribute("data-theme", theme);
}
