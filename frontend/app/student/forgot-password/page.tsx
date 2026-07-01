"use client";

import { useState } from "react";
import Link from "next/link";
import { authApi } from "@/lib/api";
import { toast } from "@/app/components/ui/Toast";

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [sent, setSent] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!email) {
      toast.error("Please enter your email address");
      return;
    }
    setLoading(true);
    try {
      await authApi.forgotPassword(email);
      setSent(true);
    } catch {
      // Always show success screen to prevent email enumeration attacks
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
      <div className="fade-in" style={{ width: "100%", maxWidth: 440 }}>
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

        <div className="glass-card" style={{ padding: 32 }}>
          {!sent ? (
            <>
              <p style={{ color: "var(--text-secondary)", fontSize: 14, marginBottom: 24, lineHeight: 1.7, textAlign: "center" }}>
                Enter the email address registered with your account. We will send you a reset link.
              </p>

              <form onSubmit={handleSubmit}>
                <div style={{ marginBottom: 20 }}>
                  <label style={{
                    display: "block", color: "var(--text-secondary)",
                    fontSize: 12, fontWeight: 600, marginBottom: 8,
                    letterSpacing: "0.05em", textTransform: "uppercase",
                  }}>
                    Email Address
                  </label>
                  <input
                    id="forgot-email-input"
                    className="ez-input"
                    type="email"
                    placeholder="your@email.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    autoFocus
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
                  {loading ? "Sending…" : "Send Reset Link →"}
                </button>
              </form>
            </>
          ) : (
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 56, marginBottom: 16 }}>📧</div>
              <h2 style={{ color: "var(--text-primary)", fontSize: 20, fontWeight: 700, marginBottom: 12 }}>
                Check Your Inbox
              </h2>
              <p style={{ color: "var(--text-secondary)", fontSize: 14, lineHeight: 1.7 }}>
                If <strong>{email}</strong> is registered, you will receive a reset link shortly. Check your spam folder too.
              </p>
              <p style={{ color: "var(--text-muted)", fontSize: 13, marginTop: 12 }}>
                The link expires in 2 hours.
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
