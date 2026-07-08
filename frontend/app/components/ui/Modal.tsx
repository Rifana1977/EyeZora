"use client";

import { useEffect, useRef } from "react";

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  footer?: React.ReactNode;
  size?: "sm" | "md" | "lg";
}

export default function Modal({
  isOpen,
  onClose,
  title,
  children,
  footer,
  size = "md",
}: ModalProps) {
  const overlayRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => { document.body.style.overflow = ""; };
  }, [isOpen]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  if (!isOpen) return null;

  const widths = { sm: 400, md: 560, lg: 760 };

  return (
    <div
      ref={overlayRef}
      onClick={(e) => { if (e.target === overlayRef.current) onClose(); }}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.7)",
        backdropFilter: "blur(8px)",
        zIndex: 1000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 20,
      }}
    >
      <div
        className="fade-in"
        style={{
          background: "var(--bg-card)",
          border: "1px solid var(--bg-border)",
          borderRadius: 20,
          width: "100%",
          maxWidth: widths[size],
          maxHeight: "90vh",
          overflow: "auto",
          boxShadow: "0 25px 60px rgba(0,0,0,0.5)",
        }}
      >
        {/* Header */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            padding: "20px 24px",
            borderBottom: "1px solid var(--bg-border)",
          }}
        >
          <h3 style={{ color: "var(--text-primary)", fontSize: 18, fontWeight: 700, margin: 0 }}>
            {title}
          </h3>
          <button
            onClick={onClose}
            style={{
              background: "rgba(255,255,255,0.08)",
              border: "none",
              color: "var(--text-secondary)",
              width: 32,
              height: 32,
              borderRadius: 8,
              cursor: "pointer",
              fontSize: 16,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              transition: "all 0.2s",
            }}
            onMouseOver={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.15)")}
            onMouseOut={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.08)")}
          >
            ✕
          </button>
        </div>

        {/* Body */}
        <div style={{ padding: "24px" }}>{children}</div>

        {/* Footer */}
        {footer && (
          <div
            style={{
              padding: "16px 24px",
              borderTop: "1px solid var(--bg-border)",
              display: "flex",
              justifyContent: "flex-end",
              gap: 12,
            }}
          >
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Confirm Dialog ─────────────────────────────────────────────────────────────

interface ConfirmProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  message: string;
  confirmLabel?: string;
  danger?: boolean;
  loading?: boolean;
}

export function ConfirmDialog({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  confirmLabel = "Confirm",
  danger = false,
  loading = false,
}: ConfirmProps) {
  return (
    <Modal
      isOpen={isOpen}
      onClose={loading ? () => {} : onClose}
      title={title}
      size="sm"
      footer={
        <>
          <style>{`
            @keyframes confirm-spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
            .confirm-spinner {
              border: 2px solid rgba(255,255,255,0.3);
              border-radius: 50%;
              border-top: 2px solid #fff;
              width: 14px;
              height: 14px;
              animation: confirm-spin 1s linear infinite;
              display: inline-block;
            }
          `}</style>
          <button
            onClick={onClose}
            disabled={loading}
            style={{
              background: "rgba(255,255,255,0.08)",
              border: "1px solid var(--bg-border)",
              color: "var(--text-secondary)",
              padding: "8px 18px",
              borderRadius: 8,
              cursor: loading ? "not-allowed" : "pointer",
              fontWeight: 500,
              opacity: loading ? 0.5 : 1,
            }}
          >
            Cancel
          </button>
          <button
            onClick={loading ? undefined : onConfirm}
            disabled={loading}
            style={{
              background: danger
                ? "linear-gradient(135deg,#b91c1c,#ef4444)"
                : "linear-gradient(135deg,#7c3aed,#5b21b6)",
              border: "none",
              color: "#fff",
              padding: "8px 18px",
              borderRadius: 8,
              cursor: loading ? "not-allowed" : "pointer",
              fontWeight: 600,
              opacity: loading ? 0.7 : 1,
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}
          >
            {loading && <span className="confirm-spinner" />}
            {loading ? "Processing..." : confirmLabel}
          </button>
        </>
      }
    >
      <p style={{ color: "var(--text-secondary)", margin: 0 }}>{message}</p>
    </Modal>
  );
}
