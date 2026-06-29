"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { authApi } from "@/lib/api";
import { saveAuth } from "@/lib/auth";
import { toast } from "@/app/components/ui/Toast";

export default function AdminLoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPw, setShowPw] = useState(false);

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault();
    if (!email || !password) {
      toast.error("Please enter email and password");
      return;
    }
    setLoading(true);
    try {
      const { token, user } = await authApi.adminLogin(email, password);
      saveAuth(token, user);

      // Set cookies for middleware route protection
      document.cookie = `ez_token=${token}; path=/; max-age=28800`;
      document.cookie = `ez_role=admin; path=/; max-age=28800`;

      toast.success(`Welcome back, ${user.name}!`);
      router.push("/admin/dashboard");
    } catch (err: any) {
      toast.error(err.message || "Login failed");
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
            Admin Control Center
          </p>
        </div>

        {/* Card */}
        <div className="glass-card" style={{ padding: 36 }}>
          {/* Title */}
          <div style={{ marginBottom: 28 }}>
            <h2 style={{ color: "var(--text-primary)", fontSize: 22, fontWeight: 700, marginBottom: 4 }}>
              Admin Login
            </h2>
            <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
              Secure access for examination administrators
            </p>
          </div>

          <form onSubmit={handleLogin}>
            {/* Email */}
            <div style={{ marginBottom: 16 }}>
              <label style={{ display: "block", color: "var(--text-secondary)", fontSize: 12, fontWeight: 600, marginBottom: 8, letterSpacing: "0.05em", textTransform: "uppercase" }}>
                Email Address
              </label>
              <input
                className="ez-input"
                type="email"
                placeholder="admin@eyezora.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="email"
              />
            </div>

            {/* Password */}
            <div style={{ marginBottom: 24 }}>
              <label style={{ display: "block", color: "var(--text-secondary)", fontSize: 12, fontWeight: 600, marginBottom: 8, letterSpacing: "0.05em", textTransform: "uppercase" }}>
                Password
              </label>
              <div style={{ position: "relative" }}>
                <input
                  className="ez-input"
                  type={showPw ? "text" : "password"}
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  style={{ paddingRight: 50 }}
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

            {/* Submit */}
            <button
              id="admin-login-btn"
              type="submit"
              disabled={loading}
              className="btn-glow"
              style={{
                width: "100%",
                padding: "13px 0",
                borderRadius: 12,
                fontSize: 15,
                fontWeight: 700,
                letterSpacing: "0.03em",
                opacity: loading ? 0.7 : 1,
              }}
            >
              {loading ? "Authenticating…" : "Login to Dashboard →"}
            </button>
          </form>

          {/* Security note */}
          <p style={{ textAlign: "center", color: "var(--text-muted)", fontSize: 12, marginTop: 20 }}>
            🔒 Secured with JWT — Admin access only
          </p>
        </div>

        {/* Student login link */}
        <p style={{ textAlign: "center", color: "var(--text-muted)", fontSize: 13, marginTop: 20 }}>
          Are you a student?{" "}
          <a href="/student/login" style={{ color: "var(--text-accent)", textDecoration: "none", fontWeight: 600 }}>
            Student Login →
          </a>
        </p>
      </div>
    </div>
  );
}
