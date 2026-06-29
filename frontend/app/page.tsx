"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { getRole } from "@/lib/auth";

export default function HomePage() {
  const router = useRouter();

  useEffect(() => {
    const role = getRole();
    if (role === "admin") router.replace("/admin/dashboard");
    else if (role === "student") router.replace("/student/pre-exam");
    else router.replace("/admin/login");
  }, [router]);

  return (
    <div style={{
      minHeight: "100vh",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      background: "var(--bg-base)",
    }}>
      <div style={{ textAlign: "center" }}>
        <div style={{
          width: 60,
          height: 60,
          borderRadius: 16,
          background: "linear-gradient(135deg,#7c3aed,#5b21b6)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 30,
          margin: "0 auto 16px",
          boxShadow: "0 8px 30px rgba(124,58,237,0.4)",
        }}>
          👁
        </div>
        <p style={{ color: "var(--text-muted)", fontSize: 14 }}>Redirecting…</p>
      </div>
    </div>
  );
}
