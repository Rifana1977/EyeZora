interface BadgeProps {
  label: string;
  variant: "low" | "medium" | "high" | "info" | "default";
  size?: "sm" | "md";
}

export function Badge({ label, variant, size = "md" }: BadgeProps) {
  const cls = `badge-${variant}`;
  const padding = size === "sm" ? "3px 10px" : "5px 14px";
  const fontSize = size === "sm" ? 11 : 12;

  return (
    <span
      className={cls}
      style={{
        display: "inline-flex",
        alignItems: "center",
        padding,
        borderRadius: 999,
        fontSize,
        fontWeight: 700,
        letterSpacing: "0.04em",
        textTransform: "uppercase",
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </span>
  );
}

export function RiskBadge({ level }: { level: "Low" | "Medium" | "High" }) {
  const map: Record<string, "low" | "medium" | "high"> = {
    Low: "low",
    Medium: "medium",
    High: "high",
  };
  const icons = { Low: "●", Medium: "●", High: "●" };
  return (
    <span
      className={`badge-${map[level]}`}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        padding: "5px 14px",
        borderRadius: 999,
        fontSize: 12,
        fontWeight: 700,
        whiteSpace: "nowrap",
      }}
    >
      <span className="pulse-dot">{icons[level]}</span>
      {level}
    </span>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const map: Record<string, { variant: "low" | "medium" | "high" | "info"; label: string }> = {
    completed: { variant: "low", label: "Completed" },
    flagged:   { variant: "high", label: "Flagged" },
    in_progress: { variant: "info", label: "In Progress" },
    abandoned: { variant: "medium", label: "Abandoned" },
  };
  const { variant, label } = map[status] || { variant: "info", label: status };
  return <Badge label={label} variant={variant} />;
}
