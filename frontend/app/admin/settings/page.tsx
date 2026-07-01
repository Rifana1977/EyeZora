"use client";

import Sidebar from "@/app/components/Sidebar";

export default function SettingsPage() {
  return (
    <div style={{ display: "flex", minHeight: "100vh" }} className="animated-bg">
      <Sidebar role="admin" />

      <main style={{ flex: 1, padding: "36px 40px", overflowY: "auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 28 }}>
          <h1 style={{ color: "var(--text-primary)", fontSize: 26, fontWeight: 800, marginBottom: 4 }}>
            System Settings & Security Docs
          </h1>
          <p style={{ color: "var(--text-muted)", fontSize: 14 }}>
            Overview of AI Proctoring configurations and browser security sandbox capabilities.
          </p>
        </div>

        {/* Configurations Card */}
        <div className="glass-card" style={{ padding: 24, marginBottom: 28 }}>
          <h2 style={{ color: "var(--text-primary)", fontSize: 18, fontWeight: 700, marginBottom: 16 }}>
            ⚙️ System Configurations
          </h2>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
            <div style={{ padding: 16, background: "var(--bg-elevated)", border: "1px solid var(--bg-border)", borderRadius: 10 }}>
              <span style={{ color: "var(--text-secondary)", fontSize: 12, fontWeight: 600, textTransform: "uppercase" }}>AI API Connection URL</span>
              <div style={{ color: "var(--text-primary)", fontSize: 15, fontWeight: 700, marginTop: 4, fontFamily: "monospace" }}>
                {process.env.NEXT_PUBLIC_AI_URL || "http://localhost:8000"}
              </div>
              <p style={{ color: "var(--text-muted)", fontSize: 11, margin: "6px 0 0" }}>
                Configured via <code style={{ background: "rgba(124,58,237,0.1)", padding: "1px 4px", borderRadius: 4 }}>NEXT_PUBLIC_AI_URL</code> environment variable.
              </p>
            </div>
            
            <div style={{ padding: 16, background: "var(--bg-elevated)", border: "1px solid var(--bg-border)", borderRadius: 10 }}>
              <span style={{ color: "var(--text-secondary)", fontSize: 12, fontWeight: 600, textTransform: "uppercase" }}>Backend API Connection URL</span>
              <div style={{ color: "var(--text-primary)", fontSize: 15, fontWeight: 700, marginTop: 4, fontFamily: "monospace" }}>
                {process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000"}
              </div>
              <p style={{ color: "var(--text-muted)", fontSize: 11, margin: "6px 0 0" }}>
                Configured via <code style={{ background: "rgba(124,58,237,0.1)", padding: "1px 4px", borderRadius: 4 }}>NEXT_PUBLIC_API_URL</code> environment variable.
              </p>
            </div>
          </div>
        </div>

        {/* Security Sandbox Limitations Documentation */}
        <div className="glass-card" style={{ padding: 28 }}>
          <h2 style={{ color: "var(--text-primary)", fontSize: 18, fontWeight: 700, marginBottom: 14 }}>
            🛡️ Browser Security Sandbox & Proctoring Capabilities
          </h2>
          <p style={{ color: "var(--text-secondary)", fontSize: 14, lineHeight: 1.7, marginBottom: 20 }}>
            EyeZora operates entirely within standard modern web browsers. Understanding browser security policies is critical for invigilators and administrators. The table below lists what events can be detected and what falls outside browser sandboxing.
          </p>

          <div style={{ overflowX: "auto" }}>
            <table className="ez-table" style={{ width: "100%", fontSize: 13 }}>
              <thead>
                <tr>
                  <th style={{ width: "25%" }}>Proctoring Metric</th>
                  <th style={{ width: "15%" }}>Status</th>
                  <th style={{ width: "60%" }}>Technical Implementation & Browser Scope</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ fontWeight: 600 }}>Tab / Window Switches</td>
                  <td><span style={{ color: "#10b981", fontWeight: 700 }}>● Monitored</span></td>
                  <td>Uses the HTML5 Page Visibility API (<code style={{ fontSize: 11 }}>document.visibilityState</code>). Logs <code style={{ fontSize: 11 }}>TAB_SWITCH</code> immediately when the student switches tabs or minimizes the window.</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>OS-Level Focus Lost</td>
                  <td><span style={{ color: "#10b981", fontWeight: 700 }}>● Monitored</span></td>
                  <td>Tracked via window blur events (<code style={{ fontSize: 11 }}>window.onblur</code> and <code style={{ fontSize: 11 }}>window.onfocus</code>). Detects when a user clicks outside the browser browser (e.g. clicking on a secondary app or OS popup). Logs <code style={{ fontSize: 11 }}>WINDOW_FOCUS_LOST</code>.</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Fullscreen Exit</td>
                  <td><span style={{ color: "#10b981", fontWeight: 700 }}>● Monitored</span></td>
                  <td>Monitored using the HTML5 Fullscreen API (<code style={{ fontSize: 11 }}>fullscreenchange</code> listener). Exiting fullscreen locks the exam screen with an overlay and logs a <code style={{ fontSize: 11 }}>FULLSCREEN_EXIT</code> violation.</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Camera Disconnection</td>
                  <td><span style={{ color: "#10b981", fontWeight: 700 }}>● Monitored</span></td>
                  <td>MediaStream track status listeners detect hardware disconnection or mute operations, triggering immediate log warnings.</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Browser Extensions</td>
                  <td><span style={{ color: "#f59e0b", fontWeight: 700 }}>● Limited</span></td>
                  <td>Webpages cannot view, disable, or modify installed extensions (Chrome/Firefox sandbox protection). We monitor unexpected document injection warnings (<code style={{ fontSize: 11 }}>EXTENSION_WARNING</code>) when third-party extension scripts write into the page DOM.</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>OS Screen-Sharing / Projections</td>
                  <td><span style={{ color: "#ef4444", fontWeight: 700 }}>✗ Blocked by Sandbox</span></td>
                  <td>Browsers restrict web apps from detecting OS-level screen sharing (e.g., HDMI mirroring, Chromecast, or OS native casting) unless explicitly sharing screen via browser request. OS level mirroring cannot be blocked.</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>External Applications</td>
                  <td><span style={{ color: "#ef4444", fontWeight: 700 }}>✗ Blocked by Sandbox</span></td>
                  <td>Webpages cannot scan running processes, check installed apps, or prevent background programs (like Discord, Teams, or screen recorders) from running. Focus blur monitoring is used to mitigate this.</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div style={{
            marginTop: 24, padding: "14px 18px", borderRadius: 10,
            background: "rgba(124,58,237,0.06)", border: "1px solid rgba(124,58,237,0.15)",
          }}>
            <p style={{ color: "var(--text-secondary)", fontSize: 13, margin: 0, lineHeight: 1.6 }}>
              💡 <strong>Best Proctoring Practices:</strong> To enforce absolute desktop security, it is recommended to run examinations in a custom <strong>Lockdown Browser</strong> shell that implements a custom CEF/Chromium wrapper disabling window transitions, system hotkeys (Alt+Tab), and secondary monitor outputs.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
