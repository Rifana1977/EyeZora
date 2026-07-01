"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { authApi } from "@/lib/api";
import { getUser, clearAuth } from "@/lib/auth";
import { toast } from "@/app/components/ui/Toast";

export default function ChangePasswordPage() {
  const router = useRouter();
  const user = getUser();
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    if (!currentPassword || !newPassword || !confirm) {
      setError("Please fill in all fields");
      return;
    }
    if (newPassword.length < 8) {
      setError("New password must be at least 8 characters");
      return;
    }
    if (newPassword !== confirm) {
      setError("New passwords do not match");
      return;
    }
    setLoading(true);
    try {
      await authApi.changePassword(currentPassword, newPassword);
      toast.success("Password changed successfully! Please log in again.");
      clearAuth();
      document.cookie = "ez_token=; Max-Age=0; path=/";
      document.cookie = "ez_role=; Max-Age=0; path=/";
      router.push("/student/login");
    } catch (err: any) {
      setError(err.message || "Failed to change password");
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
            🔒
          </div>
          <h1 style={{
            fontSize: 22, fontWeight: 800,
            background: "linear-gradient(135deg,#a78bfa,#7c3aed)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            letterSpacing: 2, marginBottom: 4,
          }}>
            CHANGE PASSWORD
          </h1>
          <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
            EyeZora AI Exam System
          </p>
        </div>

        <div className="glass-card" style={{ padding: 32 }}>
          {/* Security notice */}
          <div style={{
            background: "rgba(124,58,237,0.08)",
            border: "1px solid rgba(124,58,237,0.25)",
            borderRadius: 10,
            padding: "14px 16px",
            marginBottom: 24,
          }}>
            <p style={{ color: "var(--text-accent)", fontSize: 13, margin: 0, fontWeight: 600 }}>
              🔑 First Login Detected
            </p>
            <p style={{ color: "var(--text-secondary)", fontSize: 12, margin: "6px 0 0", lineHeight: 1.6 }}>
              For your security, you must change your temporary password before proceeding to your exam.
              {user?.name && ` Welcome, ${user.name}!`}
            </p>
          </div>

          {error && (
            <div style={{
              background: "rgba(239,68,68,0.08)",
              border: "1px solid rgba(239,68,68,0.25)",
              borderRadius: 10,
              padding: "12px 14px",
              marginBottom: 20,
            }}>
              <p style={{ color: "#ef4444", fontSize: 13, margin: 0 }}>{error}</p>
            </div>
          )}

          <form onSubmit={handleSubmit}>
            <div style={{ marginBottom: 16 }}>
              <label style={{
                display: "block", color: "var(--text-secondary)",
                fontSize: 12, fontWeight: 600, marginBottom: 8,
                letterSpacing: "0.05em", textTransform: "uppercase",
              }}>
                Current (Temporary) Password
              </label>
              <input
                id="current-password-input"
                className="ez-input"
                type="password"
                placeholder="Enter the password you received"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                autoFocus
              />
            </div>
            <div style={{ marginBottom: 16 }}>
              <label style={{
                display: "block", color: "var(--text-secondary)",
                fontSize: 12, fontWeight: 600, marginBottom: 8,
                letterSpacing: "0.05em", textTransform: "uppercase",
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
              />
            </div>
            <div style={{ marginBottom: 24 }}>
              <label style={{
                display: "block", color: "var(--text-secondary)",
                fontSize: 12, fontWeight: 600, marginBottom: 8,
                letterSpacing: "0.05em", textTransform: "uppercase",
              }}>
                Confirm New Password
              </label>
              <input
                id="confirm-password-input"
                className="ez-input"
                type="password"
                placeholder="Repeat your new password"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
              />
            </div>

            <button
              id="change-password-btn"
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
              {loading ? "Updating Password…" : "Set New Password & Continue →"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
