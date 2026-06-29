interface SkeletonProps {
  width?: string | number;
  height?: string | number;
  borderRadius?: string | number;
  className?: string;
}

export function Skeleton({ width = "100%", height = 20, borderRadius = 8, className = "" }: SkeletonProps) {
  return (
    <div
      className={`skeleton ${className}`}
      style={{ width, height, borderRadius }}
    />
  );
}

export function SkeletonCard() {
  return (
    <div style={{
      background: "var(--bg-card)",
      border: "1px solid var(--bg-border)",
      borderRadius: 16,
      padding: 24,
    }}>
      <Skeleton height={12} width="40%" borderRadius={6} />
      <div style={{ marginTop: 16 }}>
        <Skeleton height={32} width="60%" borderRadius={8} />
      </div>
    </div>
  );
}

export function SkeletonRow() {
  const widths = ["70%", "85%", "60%", "75%", "90%"];
  return (
    <tr>
      {[1, 2, 3, 4, 5].map((i, idx) => (
        <td key={i} style={{ padding: "16px 20px" }}>
          <Skeleton height={14} width={widths[idx % widths.length]} borderRadius={6} />
        </td>
      ))}
    </tr>
  );
}

export function SkeletonQuestion() {
  return (
    <div style={{
      background: "var(--bg-card)",
      border: "1px solid var(--bg-border)",
      borderRadius: 16,
      padding: 24,
      marginBottom: 16,
    }}>
      <Skeleton height={16} width="80%" borderRadius={6} />
      <div style={{ marginTop: 16, display: "flex", flexDirection: "column", gap: 10 }}>
        {[1, 2, 3, 4].map((i) => (
          <Skeleton key={i} height={12} width={`${50 + i * 8}%`} borderRadius={6} />
        ))}
      </div>
    </div>
  );
}
