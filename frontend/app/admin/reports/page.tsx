"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

/**
 * /admin/reports — redirects to /admin/monitoring
 * since reports are accessed from the monitoring table.
 */
export default function ReportsIndexPage() {
  const router = useRouter();

  useEffect(() => {
    router.replace("/admin/monitoring");
  }, [router]);

  return null;
}
