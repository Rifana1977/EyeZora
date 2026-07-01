"use client";

import { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { authApi } from "@/lib/api";
import { toast } from "@/app/components/ui/Toast";

function ResetPasswordForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const token = searchParams.get("token") || "";
  const email = searchParams.get("email") || "";

  const [newPassword, setNewPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!token || !email) {
      setError("Invalid or missing reset link. Please request a new one.");
    }
  }, [token, email]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    if (!newPassword || !confirm) {
      setError("Please fill in both fields");
      return;
    }
    if (newPassword.length < 8) {
      setError("Password must be at least 8 characters");
      return;
    }
    if (newPassword !== confirm) {
      setError("Passwords do not match");
      return;
    }
    setLoading(true);
    try {
      await authApi.resetPassword(token, email, newPassword);
      setDone(true);
      toast.success("Password reset successful!");
    } catch (err: any) {
      setError(err.message || "Reset failed. The link may have expired.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="glass-card" style={{ padding: 32 }}>
      {done ? (
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 56, marginBottom: 16 }}>✅</div>
          <h2 style={{ color: "var(--text-primary)", fontSize: 20, fontWeight: 700, marginBottom: 12 }}>
            Password Reset!
          </h2>
          <p style={{ color: "var(--text-secondary)", fontSize: 14, lineHeight: 1.7, marginBottom: 24 }}>
            Your password has been updated successfully. You can now log in with your new password.
          </p>
          <button
            onClick={() => router.push("/student/login")}
            className="btn-glow"
            style={{ padding: "12px 28px", borderRadius: 12, fontSize: 15, fontWeight: 700 }}
          >
            Go to Login →
          </button>
        </div>
      ) : (
        <>
          {error && (
            <div style={{
              background: "rgba(239,68,68,0.08)",
              border: "1px solid rgba(239,68,68,0.25)",
              borderRadius: 10,
              padding: "12px 14px",
              marginBottom: 20
            }}>
              <p style={{ color: "#ef4444", fontSize: 13, margin: 0 }}>{error}</p>
            </div>
          )}
          <form onSubmit={handleSubmit}>
            <div style={{ marginBottom: 16 }}>
              <label style={{
                display: "block",
                color: "var(--text-secondary)",
                fontSize: 12,
                fontWeight: 600,
                marginBottom: 8,
                letterSpacing: "0.05em",
                textTransform: "uppercase"
              }}>
                New Password
              </label>
              <input
                id="new-password-input"
                className="ez-input"
                type="password"
                placeholder="At least 8 characters"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                autoFocus
                disabled={!token}
              />
            </div>
            <div style={{ marginBottom: 24 }}>
              <label style={{
                display: "block",
                color: "var(--text-secondary)",
                fontSize: 12,
                fontWeight: 600,
                marginBottom: 8,
                letterSpacing: "0.05em",
                textTransform: "uppercase"
              }}>
                Confirm Password
              </label>
              <input
                id="confirm-password-input"
                className="ez-input"
                type="password"
                placeholder="Repeat your new password"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                disabled={!token}
              />
            </div>
            <button
              id="reset-password-btn"
              type="submit"
              disabled={loading || !token}
              className="btn-glow"
              style={{ width: "100%", padding: "13px 0", borderRadius: 12, fontSize: 15, fontWeight: 700 }}
            >
              {loading ? "Resetting…" : "Set New Password →"}
            </button>
          </form>
        </>
      )}
      <div style={{ marginTop: 24, textAlign: "center" }}>
        <Link href="/student/login" style={{ color: "var(--text-muted)", fontSize: 13, textDecoration: "none" }}>
          ← Back to Login
        </Link>
      </div>
    </div>
  );
}

export default function ResetPasswordPage() {
  return (
    <div className="animated-bg" style={{
      minHeight: "100vh",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      padding: 20,
    }}>
      <div className="fade-in" style={{ width: "100%", maxWidth: 440 }}>
        {/* Brand */}
        <div style={{ textAlign: "center", marginBottom: 36 }}>
          <div style={{
            width: 64, height: 64, borderRadius: 18,
            background: "linear-gradient(135deg,#7c3aed,#5b21b6)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 32, margin: "0 auto 16px",
            boxShadow: "0 8px 30px rgba(124,58,237,0.45)"
          }}>
            🔑
          </div>
          <h1 style={{
            fontSize: 24, fontWeight: 800,
            background: "linear-gradient(135deg,#a78bfa,#7c3aed)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            letterSpacing: 2, marginBottom: 4
          }}>
            RESET PASSWORD
          </h1>
          <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
            EyeZora AI Exam System
          </p>
        </div>

        <Suspense fallback={
          <div className="glass-card" style={{ padding: 48, textAlign: "center" }}>
            <div style={{
              width: 36, height: 36,
              border: "3px solid rgba(124,58,237,0.3)",
              borderTopColor: "#7c3aed",
              borderRadius: "50%",
              animation: "spin 0.8s linear infinite",
              margin: "0 auto 12px"
            }} />
            <p style={{ color: "var(--text-secondary)", fontSize: 14 }}>Loading reset parameters…</p>
            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
          </div>
        }>
          <ResetPasswordForm />
        </Suspense>
      </div>
    </div>
  );
}
