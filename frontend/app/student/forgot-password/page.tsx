"use client";

import { useState } from "react";
import Link from "next/link";
import { authApi } from "@/lib/api";
import { toast } from "@/app/components/ui/Toast";

export default function ForgotPasswordPage() {
  const [studentId, setStudentId] = useState("");
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [sent, setSent] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!studentId.trim()) {
      toast.error("Please enter your Student ID");
      return;
    }
    if (!email.trim()) {
      toast.error("Please enter your registered email address");
      return;
    }
    setLoading(true);
    try {
      await authApi.forgotPassword(studentId.trim(), email.trim());
      setSent(true);
    } catch {
      // Always show success screen — never reveal whether details matched.
      // This prevents user enumeration attacks.
      setSent(true);
    } finally {
      setLoading(false);
    }
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
            🔐
          </div>
          <h1 style={{
            fontSize: 26, fontWeight: 800,
            background: "linear-gradient(135deg,#a78bfa,#7c3aed,#c084fc)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            letterSpacing: 2, marginBottom: 4,
          }}>
            FORGOT PASSWORD
          </h1>
          <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
            EyeZora AI Exam System
          </p>
        </div>

        <div className="glass-card" style={{ padding: 36 }}>
          {!sent ? (
            <>
              {/* Info box */}
              <div style={{
                background: "rgba(124,58,237,0.07)",
                border: "1px solid rgba(124,58,237,0.2)",
                borderRadius: 10,
                padding: "12px 16px",
                marginBottom: 28,
              }}>
                <p style={{ color: "var(--text-secondary)", fontSize: 13, margin: 0, lineHeight: 1.65 }}>
                  🔒 Enter your <strong>Student ID</strong> and <strong>registered email address</strong>.
                  Both must match the same account before a reset link is sent.
                </p>
              </div>

              <form onSubmit={handleSubmit}>
                {/* Student ID field */}
                <div style={{ marginBottom: 18 }}>
                  <label style={{
                    display: "block", color: "var(--text-secondary)",
                    fontSize: 12, fontWeight: 600, marginBottom: 8,
                    letterSpacing: "0.05em", textTransform: "uppercase",
                  }}>
                    Student ID
                  </label>
                  <input
                    id="forgot-student-id-input"
                    className="ez-input"
                    type="text"
                    placeholder="e.g. STU001"
                    value={studentId}
                    onChange={(e) => setStudentId(e.target.value)}
                    autoFocus
                    autoComplete="username"
                  />
                </div>

                {/* Email field */}
                <div style={{ marginBottom: 28 }}>
                  <label style={{
                    display: "block", color: "var(--text-secondary)",
                    fontSize: 12, fontWeight: 600, marginBottom: 8,
                    letterSpacing: "0.05em", textTransform: "uppercase",
                  }}>
                    Registered Email Address
                  </label>
                  <input
                    id="forgot-email-input"
                    className="ez-input"
                    type="email"
                    placeholder="your@email.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    autoComplete="email"
                  />
                </div>

                <button
                  id="send-reset-btn"
                  type="submit"
                  disabled={loading}
                  className="btn-glow"
                  style={{
                    width: "100%",
                    padding: "13px 0",
                    borderRadius: 12,
                    fontSize: 15,
                    fontWeight: 700,
                  }}
                >
                  {loading ? "Sending…" : "Send Password Reset Link →"}
                </button>
              </form>
            </>
          ) : (
            /* Success screen — generic, regardless of match */
            <div style={{ textAlign: "center", padding: "8px 0" }}>
              <div style={{ fontSize: 56, marginBottom: 16 }}>📧</div>
              <h2 style={{
                color: "var(--text-primary)", fontSize: 20,
                fontWeight: 700, marginBottom: 12,
              }}>
                Check Your Inbox
              </h2>
              <p style={{ color: "var(--text-secondary)", fontSize: 14, lineHeight: 1.7, marginBottom: 12 }}>
                If the details you provided are valid, you will receive a
                password reset link shortly. Please check your spam folder too.
              </p>
              <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
                The link expires in <strong>15 minutes</strong>.
              </p>
            </div>
          )}

          <div style={{ marginTop: 24, textAlign: "center" }}>
            <Link href="/student/login" style={{
              color: "var(--text-muted)", fontSize: 13, textDecoration: "none",
              display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
            }}>
              ← Back to Login
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
