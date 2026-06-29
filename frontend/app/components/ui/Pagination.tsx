interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}

export default function Pagination({ currentPage, totalPages, onPageChange }: PaginationProps) {
  if (totalPages <= 1) return null;

  const pages: (number | "...")[] = [];

  if (totalPages <= 7) {
    for (let i = 1; i <= totalPages; i++) pages.push(i);
  } else {
    pages.push(1);
    if (currentPage > 3) pages.push("...");
    for (let i = Math.max(2, currentPage - 1); i <= Math.min(totalPages - 1, currentPage + 1); i++) {
      pages.push(i);
    }
    if (currentPage < totalPages - 2) pages.push("...");
    pages.push(totalPages);
  }

  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8, padding: "16px 0" }}>
      <PageBtn
        label="‹"
        disabled={currentPage === 1}
        onClick={() => onPageChange(currentPage - 1)}
      />

      {pages.map((p, i) =>
        p === "..." ? (
          <span key={`ellipsis-${i}`} style={{ color: "var(--text-muted)", padding: "0 4px" }}>
            …
          </span>
        ) : (
          <PageBtn
            key={p}
            label={String(p)}
            active={p === currentPage}
            onClick={() => onPageChange(p as number)}
          />
        )
      )}

      <PageBtn
        label="›"
        disabled={currentPage === totalPages}
        onClick={() => onPageChange(currentPage + 1)}
      />
    </div>
  );
}

function PageBtn({
  label,
  active = false,
  disabled = false,
  onClick,
}: {
  label: string;
  active?: boolean;
  disabled?: boolean;
  onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        width: 36,
        height: 36,
        borderRadius: 8,
        border: active
          ? "1px solid var(--brand-primary)"
          : "1px solid var(--bg-border)",
        background: active
          ? "var(--brand-primary)"
          : disabled
          ? "transparent"
          : "rgba(255,255,255,0.04)",
        color: active ? "#fff" : disabled ? "var(--text-muted)" : "var(--text-secondary)",
        cursor: disabled ? "not-allowed" : "pointer",
        fontWeight: active ? 700 : 500,
        fontSize: 14,
        transition: "all 0.15s",
      }}
    >
      {label}
    </button>
  );
}
