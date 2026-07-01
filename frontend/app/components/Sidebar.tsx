"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { clearAuth, getTheme, setTheme, type Theme } from "@/lib/auth";
import { useState, useEffect } from "react";

const adminLinks = [
  { label: "Dashboard", href: "/admin/dashboard", icon: "⊞" },
  { label: "Students", href: "/admin/students", icon: "👤" },
  { label: "Tests", href: "/admin/questions", icon: "📝" },
  { label: "Assign Exams", href: "/admin/assign-exams", icon: "📋" },
  { label: "Monitoring", href: "/admin/monitoring", icon: "◉" },
  { label: "Reports", href: "/admin/reports", icon: "📊" },
  { label: "Settings", href: "/admin/settings", icon: "⚙" },
];

const studentLinks = [
  { label: "My Exam", href: "/student/exam", icon: "✎" },
  { label: "My Results", href: "/student/results", icon: "📊" },
];

export default function Sidebar({ role }: { role: "student" | "admin" }) {
  const pathname = usePathname();
  const router = useRouter();
  const links = role === "admin" ? adminLinks : studentLinks;
  const [theme, setThemeState] = useState<Theme>("dark");

  useEffect(() => {
    const t = getTheme();
    setThemeState(t);
    document.documentElement.setAttribute("data-theme", t);
  }, []);

  function toggleTheme() {
    const next: Theme = theme === "dark" ? "light" : "dark";
    setTheme(next);
    setThemeState(next);
  }

  function logout() {
    clearAuth();
    document.cookie = "ez_token=; Max-Age=0; path=/";
    document.cookie = "ez_role=; Max-Age=0; path=/";
    router.push(role === "admin" ? "/admin/login" : "/student/login");
  }

  return (
    <aside style={{
      width: 260,
      minHeight: "100vh",
      background: "var(--sidebar-bg)",
      borderRight: "1px solid var(--bg-border)",
      display: "flex",
      flexDirection: "column",
      padding: "0 0 20px",
      flexShrink: 0,
    }}>
      {/* Brand */}
      <div style={{
        padding: "28px 24px 24px",
        borderBottom: "1px solid var(--bg-border)",
        marginBottom: 12,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
          <div style={{
            width: 36,
            height: 36,
            borderRadius: 10,
            background: "linear-gradient(135deg,#7c3aed,#5b21b6)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 18,
            boxShadow: "0 4px 15px rgba(124,58,237,0.4)",
          }}>
            👁
          </div>
          <span style={{
            fontSize: 20,
            fontWeight: 800,
            background: "linear-gradient(135deg,#a78bfa,#7c3aed)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            letterSpacing: 2,
          }}>
            EYEZORA
          </span>
        </div>
        <div style={{
          display: "inline-block",
          padding: "2px 10px",
          borderRadius: 999,
          background: "rgba(124,58,237,0.15)",
          border: "1px solid rgba(124,58,237,0.3)",
          fontSize: 11,
          color: "#a78bfa",
          fontWeight: 600,
          letterSpacing: 1,
          textTransform: "uppercase",
        }}>
          {role}
        </div>
      </div>

      {/* Nav */}
      <nav style={{ flex: 1, padding: "8px 12px" }}>
        {links.map((link) => {
          const active = pathname === link.href || pathname.startsWith(link.href + "/");
          return (
            <Link
              key={link.href}
              href={link.href}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                padding: "11px 14px",
                borderRadius: 10,
                marginBottom: 4,
                textDecoration: "none",
                fontSize: 14,
                fontWeight: active ? 600 : 500,
                color: active ? "#a78bfa" : "var(--nav-text)",
                background: active
                  ? "linear-gradient(135deg,rgba(124,58,237,0.18),rgba(91,33,182,0.1))"
                  : "transparent",
                borderLeft: active ? "3px solid #7c3aed" : "3px solid transparent",
                transition: "all 0.15s",
              }}
            >
              <span style={{ fontSize: 16, opacity: active ? 1 : 0.6 }}>{link.icon}</span>
              {link.label}
            </Link>
          );
        })}
      </nav>

      {/* Bottom controls */}
      <div style={{ padding: "0 12px", display: "flex", flexDirection: "column", gap: 8 }}>
        {/* Theme Toggle */}
        <button
          onClick={toggleTheme}
          id="theme-toggle-btn"
          title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          style={{
            width: "100%",
            padding: "10px 14px",
            borderRadius: 10,
            border: "1px solid var(--bg-border)",
            background: "var(--bg-elevated)",
            color: "var(--text-secondary)",
            fontSize: 13,
            fontWeight: 500,
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: 8,
            transition: "all 0.15s",
          }}
          onMouseOver={(e) => (e.currentTarget.style.background = "var(--bg-hover)")}
          onMouseOut={(e) => (e.currentTarget.style.background = "var(--bg-elevated)")}
        >
          <span>{theme === "dark" ? "☀️" : "🌙"}</span>
          {theme === "dark" ? "Light Mode" : "Dark Mode"}
        </button>

        {/* Logout */}
        <button
          onClick={logout}
          style={{
            width: "100%",
            padding: "11px 14px",
            borderRadius: 10,
            border: "1px solid rgba(239,68,68,0.25)",
            background: "rgba(239,68,68,0.06)",
            color: "#ef4444",
            fontSize: 14,
            fontWeight: 500,
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: 10,
            transition: "all 0.15s",
          }}
          onMouseOver={(e) => (e.currentTarget.style.background = "rgba(239,68,68,0.12)")}
          onMouseOut={(e) => (e.currentTarget.style.background = "rgba(239,68,68,0.06)")}
        >
          <span>⇤</span>
          Logout
        </button>
      </div>
    </aside>
  );
}
