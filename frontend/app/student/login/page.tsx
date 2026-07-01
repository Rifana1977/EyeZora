"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { authApi } from "@/lib/api";
import { saveAuth } from "@/lib/auth";
import { toast } from "@/app/components/ui/Toast";

export default function StudentLoginPage() {
  const router = useRouter();
  const [identifier, setIdentifier] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPw, setShowPw] = useState(false);

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault();
    if (!identifier || !password) {
      toast.error("Please enter your Student ID / Email and password");
      return;
    }
    setLoading(true);
    try {
      const { token, user } = await authApi.studentLogin(identifier, password);
      saveAuth(token, user);

      // Set cookies for middleware
      document.cookie = `ez_token=${token}; path=/; max-age=14400`;
      document.cookie = `ez_role=student; path=/; max-age=14400`;

      toast.success(`Welcome, ${user.name}!`);

      // If this is a temporary/first-time password, force change
      if (user.isTemporaryPassword) {
        router.push("/student/change-password");
      } else {
        // Always go to pre-exam page — it handles all assignment states
        router.push("/student/pre-exam");
      }
    } catch (err: any) {
      toast.error(err.message || "Invalid credentials");
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
            width: 64,
            height: 64,
            borderRadius: 18,
            background: "linear-gradient(135deg,#7c3aed,#5b21b6)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 32,
            margin: "0 auto 16px",
            boxShadow: "0 8px 30px rgba(124,58,237,0.45)",
          }}>
            👁
          </div>
          <h1 style={{
            fontSize: 28,
            fontWeight: 800,
            background: "linear-gradient(135deg,#a78bfa,#7c3aed,#c084fc)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            letterSpacing: 2,
            marginBottom: 4,
          }}>
            EYEZORA
          </h1>
          <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
            AI-Powered Examination System
          </p>
        </div>

        {/* Card */}
        <div className="glass-card" style={{ padding: 36 }}>
          <div style={{ marginBottom: 28 }}>
            <h2 style={{ color: "var(--text-primary)", fontSize: 22, fontWeight: 700, marginBottom: 4 }}>
              Student Login
            </h2>
            <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
              Enter your credentials to access your examination
            </p>
          </div>

          <form onSubmit={handleLogin}>
            <div style={{ marginBottom: 16 }}>
              <label style={{
                display: "block", color: "var(--text-secondary)",
                fontSize: 12, fontWeight: 600, marginBottom: 8,
                letterSpacing: "0.05em", textTransform: "uppercase",
              }}>
                Student ID or Email
              </label>
              <input
                id="student-identifier"
                className="ez-input"
                type="text"
                placeholder="e.g. STU001 or john@email.com"
                value={identifier}
                onChange={(e) => setIdentifier(e.target.value)}
                autoComplete="username"
              />
            </div>

            <div style={{ marginBottom: 24 }}>
              <label style={{
                display: "block", color: "var(--text-secondary)",
                fontSize: 12, fontWeight: 600, marginBottom: 8,
                letterSpacing: "0.05em", textTransform: "uppercase",
              }}>
                Password
              </label>
              <div style={{ position: "relative" }}>
                <input
                  id="student-password"
                  className="ez-input"
                  type={showPw ? "text" : "password"}
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  style={{ paddingRight: 50 }}
                  autoComplete="current-password"
                />
                <button
                  type="button"
                  onClick={() => setShowPw(!showPw)}
                  style={{
                    position: "absolute",
                    right: 14,
                    top: "50%",
                    transform: "translateY(-50%)",
                    background: "none",
                    border: "none",
                    color: "var(--text-muted)",
                    cursor: "pointer",
                    fontSize: 16,
                  }}
                >
                  {showPw ? "🙈" : "👁"}
                </button>
              </div>
            </div>
            {/* Forgot password link */}
            <div style={{ textAlign: "right", marginTop: -4, marginBottom: 4 }}>
              <a
                href="/student/forgot-password"
                style={{ color: "var(--text-accent)", fontSize: 12, fontWeight: 600, textDecoration: "none" }}
              >
                Forgot Password?
              </a>
            </div>

            <button
              id="student-login-btn"
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
              {loading ? "Authenticating…" : "Continue →"}
            </button>
          </form>

          {/* Instructions */}
          <div style={{
            marginTop: 24,
            padding: 14,
            background: "rgba(124,58,237,0.06)",
            border: "1px solid rgba(124,58,237,0.2)",
            borderRadius: 10,
          }}>
            <p style={{ color: "var(--text-secondary)", fontSize: 12, margin: 0, lineHeight: 1.6 }}>
              ℹ️ Your credentials were provided by your administrator. Contact your exam invigilator if you cannot login.
            </p>
          </div>
        </div>

        <p style={{ textAlign: "center", color: "var(--text-muted)", fontSize: 13, marginTop: 20 }}>
          Are you an admin?{" "}
          <a href="/admin/login" style={{ color: "var(--text-accent)", textDecoration: "none", fontWeight: 600 }}>
            Admin Login →
          </a>
        </p>
      </div>
    </div>
  );
}
