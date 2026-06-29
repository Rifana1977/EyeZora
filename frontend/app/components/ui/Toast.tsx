"use client";

import { useEffect, useState } from "react";

export type ToastType = "success" | "error" | "warning" | "info";

export interface ToastMessage {
  id: string;
  message: string;
  type: ToastType;
}

// ── Toast Store (simple module-level state) ───────────────────────────────────
let _listeners: ((toasts: ToastMessage[]) => void)[] = [];
let _toasts: ToastMessage[] = [];

function notify(toasts: ToastMessage[]) {
  _toasts = toasts;
  _listeners.forEach((fn) => fn(toasts));
}

export function toast(message: string, type: ToastType = "info") {
  const id = Math.random().toString(36).slice(2);
  notify([..._toasts, { id, message, type }]);
  setTimeout(() => {
    notify(_toasts.filter((t) => t.id !== id));
  }, 4000);
}

toast.success = (msg: string) => toast(msg, "success");
toast.error   = (msg: string) => toast(msg, "error");
toast.warning = (msg: string) => toast(msg, "warning");
toast.info    = (msg: string) => toast(msg, "info");

// ── Toast Container Component ─────────────────────────────────────────────────

const ICONS: Record<ToastType, string> = {
  success: "✓",
  error: "✕",
  warning: "⚠",
  info: "ℹ",
};

const COLORS: Record<ToastType, { bg: string; border: string; icon: string }> = {
  success: { bg: "rgba(16,185,129,0.12)", border: "rgba(16,185,129,0.35)", icon: "#10b981" },
  error:   { bg: "rgba(239,68,68,0.12)",  border: "rgba(239,68,68,0.35)",  icon: "#ef4444" },
  warning: { bg: "rgba(245,158,11,0.12)", border: "rgba(245,158,11,0.35)", icon: "#f59e0b" },
  info:    { bg: "rgba(59,130,246,0.12)", border: "rgba(59,130,246,0.35)", icon: "#3b82f6" },
};

export default function ToastContainer() {
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  useEffect(() => {
    const listener = (t: ToastMessage[]) => setToasts([...t]);
    _listeners.push(listener);
    return () => {
      _listeners = _listeners.filter((l) => l !== listener);
    };
  }, []);

  if (toasts.length === 0) return null;

  return (
    <div
      style={{
        position: "fixed",
        top: 20,
        right: 20,
        zIndex: 9999,
        display: "flex",
        flexDirection: "column",
        gap: 10,
      }}
    >
      {toasts.map((t) => {
        const c = COLORS[t.type];
        return (
          <div
            key={t.id}
            className="toast-enter"
            style={{
              display: "flex",
              alignItems: "center",
              gap: 12,
              background: c.bg,
              border: `1px solid ${c.border}`,
              backdropFilter: "blur(12px)",
              borderRadius: 12,
              padding: "14px 20px",
              minWidth: 280,
              maxWidth: 400,
              boxShadow: "0 4px 20px rgba(0,0,0,0.3)",
            }}
          >
            <span
              style={{
                width: 24,
                height: 24,
                borderRadius: "50%",
                background: c.icon,
                color: "#fff",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 13,
                fontWeight: 700,
                flexShrink: 0,
              }}
            >
              {ICONS[t.type]}
            </span>
            <p style={{ color: "#f0f2ff", fontSize: 14, fontWeight: 500, margin: 0 }}>
              {t.message}
            </p>
          </div>
        );
      })}
    </div>
  );
}
