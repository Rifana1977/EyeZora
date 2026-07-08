"use client";

import { useState, useEffect } from "react";
import { useRouter, useParams } from "next/navigation";
import Link from "next/link";
import { authApi } from "@/lib/api";
import { toast } from "@/app/components/ui/Toast";

export default function ResetPasswordPage() {
  const router = useRouter();
  const params = useParams();
  // The dynamic segment is [token] — always a string when present
  const token = Array.isArray(params.token) ? params.token[0] : params.token ?? "";

  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState("");

  // Password strength helpers
  const hasMinLength = password.length >= 8;
  const hasUppercase = /[A-Z]/.test(password);
  const hasNumber = /\d/.test(password);
  const passwordStrong = hasMinLength && hasUppercase && hasNumber;
  const passwordsMatch = password === confirm && confirm.length > 0;

  useEffect(() => {
    if (!token) {
      setError("Invalid or missing reset link. Please request a new one.");
    }
  }, [token]);

  // Redirect to login after successful reset (3-second delay)
  useEffect(() => {
    if (done) {
      const timer = setTimeout(() => router.push("/student/login"), 3000);
      return () => clearTimeout(timer);
    }
  }, [done, router]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");

    if (!password || !confirm) {
      setError("Please fill in both password fields.");
      return;
    }
    if (!hasMinLength) {
      setError("Password must be at least 8 characters.");
      return;
    }
    if (!passwordStrong) {
      setError("Password must contain at least one uppercase letter and one number.");
      return;
    }
    if (password !== confirm) {
      setError("Passwords do not match.");
      return;
    }

    setLoading(true);
    try {
      await authApi.resetPassword(token, password);
      setDone(true);
      toast.success("Password reset successful!");
    } catch (err: any) {
      setError(err.message || "Reset failed. The link may have expired.");
    } finally {
      setLoading(false);
    }
  }

  function handleCancel() {
    // Navigate back to login without changing anything
    router.push("/student/login");
  }

  return (
    <div className="animated-bg" style={{
      minHeight: "100vh",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      padding: 20,
    }}>
      <div className="fade-in" style={{ width: "100%", maxWidth: 460 }}>
        {/* Brand */}
        <div style={{ textAlign: "center", marginBottom: 36 }}>
          <div style={{
            width: 64, height: 64, borderRadius: 18,
            background: "linear-gradient(135deg,#7c3aed,#5b21b6)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 32, margin: "0 auto 16px",
            boxShadow: "0 8px 30px rgba(124,58,237,0.45)",
          }}>
            🔑
          </div>
          <h1 style={{
            fontSize: 26, fontWeight: 800,
            background: "linear-gradient(135deg,#a78bfa,#7c3aed,#c084fc)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            letterSpacing: 2, marginBottom: 4,
          }}>
            RESET PASSWORD
          </h1>
          <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
            EyeZora AI Exam System
          </p>
        </div>

        <div className="glass-card" style={{ padding: 36 }}>
          {done ? (
            /* ── Success State ─────────────────────────────────────────────── */
            <div style={{ textAlign: "center", padding: "8px 0" }}>
              <div style={{ fontSize: 56, marginBottom: 16 }}>✅</div>
              <h2 style={{
                color: "var(--text-primary)", fontSize: 20,
                fontWeight: 700, marginBottom: 12,
              }}>
                Password Reset!
              </h2>
              <p style={{
                color: "var(--text-secondary)", fontSize: 14,
                lineHeight: 1.7, marginBottom: 8,
              }}>
                Your password has been updated successfully.
                You can now log in with your new password.
              </p>
              <p style={{ color: "var(--text-muted)", fontSize: 13, marginBottom: 28 }}>
                Redirecting to login in a moment…
              </p>
              <button
                id="go-to-login-btn"
                onClick={() => router.push("/student/login")}
                className="btn-glow"
                style={{ padding: "12px 32px", borderRadius: 12, fontSize: 15, fontWeight: 700 }}
              >
                Go to Login →
              </button>
            </div>
          ) : (
            /* ── Reset Form ────────────────────────────────────────────────── */
            <>
              {/* Error banner */}
              {error && (
                <div style={{
                  background: "rgba(239,68,68,0.08)",
                  border: "1px solid rgba(239,68,68,0.25)",
                  borderRadius: 10,
                  padding: "12px 14px",
                  marginBottom: 20,
                }}>
                  <p style={{ color: "#ef4444", fontSize: 13, margin: 0 }}>⚠️ {error}</p>
                </div>
              )}

              <p style={{
                color: "var(--text-secondary)", fontSize: 14,
                lineHeight: 1.65, marginBottom: 24,
              }}>
                Choose a strong new password for your EyeZora account.
              </p>

              <form onSubmit={handleSubmit}>
                {/* New Password */}
                <div style={{ marginBottom: 18 }}>
                  <label style={{
                    display: "block", color: "var(--text-secondary)",
                    fontSize: 12, fontWeight: 600, marginBottom: 8,
                    letterSpacing: "0.05em", textTransform: "uppercase",
                  }}>
                    New Password
                  </label>
                  <div style={{ position: "relative" }}>
                    <input
                      id="new-password-input"
                      className="ez-input"
                      type={showPw ? "text" : "password"}
                      placeholder="At least 8 characters"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      disabled={!token || loading}
                      style={{ paddingRight: 48 }}
                      autoFocus
                      autoComplete="new-password"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPw(!showPw)}
                      style={{
                        position: "absolute", right: 14, top: "50%",
                        transform: "translateY(-50%)", background: "none",
                        border: "none", color: "var(--text-muted)",
                        cursor: "pointer", fontSize: 16,
                      }}
                    >
                      {showPw ? "🙈" : "👁"}
                    </button>
                  </div>

                  {/* Inline strength indicators */}
                  {password.length > 0 && (
                    <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 4 }}>
                      <StrengthHint met={hasMinLength} label="At least 8 characters" />
                      <StrengthHint met={hasUppercase} label="At least one uppercase letter" />
                      <StrengthHint met={hasNumber} label="At least one number" />
                    </div>
                  )}
                </div>

                {/* Confirm Password */}
                <div style={{ marginBottom: 28 }}>
                  <label style={{
                    display: "block", color: "var(--text-secondary)",
                    fontSize: 12, fontWeight: 600, marginBottom: 8,
                    letterSpacing: "0.05em", textTransform: "uppercase",
                  }}>
                    Confirm Password
                  </label>
                  <div style={{ position: "relative" }}>
                    <input
                      id="confirm-password-input"
                      className="ez-input"
                      type={showConfirm ? "text" : "password"}
                      placeholder="Repeat your new password"
                      value={confirm}
                      onChange={(e) => setConfirm(e.target.value)}
                      disabled={!token || loading}
                      style={{ paddingRight: 48 }}
                      autoComplete="new-password"
                    />
                    <button
                      type="button"
                      onClick={() => setShowConfirm(!showConfirm)}
                      style={{
                        position: "absolute", right: 14, top: "50%",
                        transform: "translateY(-50%)", background: "none",
                        border: "none", color: "var(--text-muted)",
                        cursor: "pointer", fontSize: 16,
                      }}
                    >
                      {showConfirm ? "🙈" : "👁"}
                    </button>
                  </div>

                  {/* Match indicator */}
                  {confirm.length > 0 && (
                    <p style={{
                      marginTop: 8, fontSize: 12, fontWeight: 500,
                      color: passwordsMatch ? "#10b981" : "#ef4444",
                    }}>
                      {passwordsMatch ? "✓ Passwords match" : "✗ Passwords do not match"}
                    </p>
                  )}
                </div>

                {/* Primary action */}
                <button
                  id="reset-password-btn"
                  type="submit"
                  disabled={loading || !token}
                  className="btn-glow"
                  style={{
                    width: "100%", padding: "13px 0",
                    borderRadius: 12, fontSize: 15, fontWeight: 700,
                    marginBottom: 12,
                  }}
                >
                  {loading ? "Resetting…" : "Reset Password →"}
                </button>

                {/* Cancel — returns to login without touching the password */}
                <button
                  id="cancel-reset-btn"
                  type="button"
                  onClick={handleCancel}
                  disabled={loading}
                  style={{
                    width: "100%", padding: "12px 0",
                    borderRadius: 12, fontSize: 14, fontWeight: 600,
                    background: "transparent",
                    border: "1px solid rgba(124,58,237,0.25)",
                    color: "var(--text-muted)",
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                  }}
                  onMouseEnter={(e) => {
                    (e.currentTarget as HTMLButtonElement).style.borderColor = "rgba(124,58,237,0.5)";
                    (e.currentTarget as HTMLButtonElement).style.color = "var(--text-secondary)";
                  }}
                  onMouseLeave={(e) => {
                    (e.currentTarget as HTMLButtonElement).style.borderColor = "rgba(124,58,237,0.25)";
                    (e.currentTarget as HTMLButtonElement).style.color = "var(--text-muted)";
                  }}
                >
                  Cancel — Back to Login
                </button>
              </form>
            </>
          )}

          {!done && (
            <div style={{ marginTop: 20, textAlign: "center" }}>
              <Link href="/student/login" style={{
                color: "var(--text-muted)", fontSize: 13, textDecoration: "none",
              }}>
                ← Back to Login
              </Link>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ── Helper: inline strength hint chip ──────────────────────────────────────── */
function StrengthHint({ met, label }: { met: boolean; label: string }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 6,
      fontSize: 12, fontWeight: 500,
      color: met ? "#10b981" : "var(--text-muted)",
    }}>
      {met ? "✓" : "○"} {label}
    </span>
  );
}
