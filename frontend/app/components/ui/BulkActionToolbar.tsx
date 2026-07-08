"use client";

interface BulkActionToolbarProps {
  count: number;
  onDelete: () => void;
  onClear: () => void;
  deleting?: boolean;
  /** Label for the delete button (default: "Delete Selected") */
  deleteLabel?: string;
  /** Confirmation noun used in the default label (default: "records") */
  noun?: string;
  /** Additional action buttons rendered before the divider */
  extraActions?: React.ReactNode;
}

/**
 * BulkActionToolbar
 *
 * Shown above a table when one or more rows are selected.
 * Provides count, optional extra actions (e.g. Publish), Delete, and Clear.
 *
 * Example:
 *   <BulkActionToolbar
 *     count={sel.selectedCount}
 *     onDelete={handleBulkDelete}
 *     onClear={sel.clearSelection}
 *     noun="students"
 *   />
 */
export function BulkActionToolbar({
  count,
  onDelete,
  onClear,
  deleting = false,
  deleteLabel,
  noun = "records",
  extraActions,
}: BulkActionToolbarProps) {
  if (count === 0) return null;

  const resolvedLabel = deleteLabel ?? `🗑 Delete Selected`;

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        padding: "12px 18px",
        marginBottom: 12,
        borderRadius: 12,
        background: "rgba(124,58,237,0.08)",
        border: "1px solid rgba(124,58,237,0.25)",
        flexWrap: "wrap",
        animation: "fadeIn 0.18s ease",
      }}
    >
      {/* Count badge */}
      <span
        style={{
          fontWeight: 700,
          fontSize: 14,
          color: "#a78bfa",
          marginRight: 4,
          whiteSpace: "nowrap",
        }}
      >
        {count} {count === 1 ? noun.replace(/s$/, "") : noun} selected
      </span>

      {/* Divider */}
      {extraActions && (
        <>
          <div
            style={{
              width: 1,
              height: 20,
              background: "rgba(124,58,237,0.25)",
              flexShrink: 0,
            }}
          />
          {extraActions}
        </>
      )}

      {/* Divider before delete */}
      <div
        style={{
          width: 1,
          height: 20,
          background: "rgba(124,58,237,0.25)",
          flexShrink: 0,
        }}
      />

      {/* Delete button */}
      <button
        onClick={onDelete}
        disabled={deleting}
        style={{
          padding: "7px 16px",
          borderRadius: 8,
          fontSize: 13,
          fontWeight: 600,
          background: "rgba(239,68,68,0.1)",
          border: "1px solid rgba(239,68,68,0.3)",
          color: "#ef4444",
          cursor: deleting ? "default" : "pointer",
          opacity: deleting ? 0.6 : 1,
          transition: "all 0.15s",
          whiteSpace: "nowrap",
        }}
        onMouseOver={(e) => {
          if (!deleting) e.currentTarget.style.background = "rgba(239,68,68,0.18)";
        }}
        onMouseOut={(e) => {
          e.currentTarget.style.background = "rgba(239,68,68,0.1)";
        }}
      >
        {deleting ? "Deleting…" : resolvedLabel}
      </button>

      {/* Clear button */}
      <button
        onClick={onClear}
        disabled={deleting}
        style={{
          padding: "7px 14px",
          borderRadius: 8,
          fontSize: 13,
          fontWeight: 500,
          background: "transparent",
          border: "1px solid rgba(124,58,237,0.2)",
          color: "var(--text-muted)",
          cursor: deleting ? "default" : "pointer",
          transition: "all 0.15s",
          whiteSpace: "nowrap",
        }}
        onMouseOver={(e) => {
          if (!deleting) {
            e.currentTarget.style.borderColor = "rgba(124,58,237,0.4)";
            e.currentTarget.style.color = "var(--text-secondary)";
          }
        }}
        onMouseOut={(e) => {
          e.currentTarget.style.borderColor = "rgba(124,58,237,0.2)";
          e.currentTarget.style.color = "var(--text-muted)";
        }}
      >
        ✕ Clear Selection
      </button>
    </div>
  );
}
