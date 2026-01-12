"use client";

import { useRouter, usePathname } from "next/navigation";

export default function Sidebar({
  role,
}: {
  role: "student" | "admin";
}) {
  const router = useRouter();
  const pathname = usePathname();

  const isActive = (path: string) => pathname === path;

  return (
    <aside className="w-72 min-h-screen bg-gradient-to-b from-black via-[#0a1633] to-black px-6 py-8">
      <h1 className="text-2xl font-extrabold text-white mb-12">EYEZORA</h1>

      {role === "admin" && (
        <>
          <NavItem
            label="Dashboard"
            active={isActive("/dashboard/admin")}
            onClick={() => router.push("/dashboard/admin")}
          />
          <NavItem
            label="Create Test"
            active={isActive("/dashboard/admin/questions")}
            onClick={() => router.push("/dashboard/admin/questions")}
          />
        </>
      )}
    </aside>
  );
}

function NavItem({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <div
      onClick={onClick}
      className={`px-4 py-3 rounded-xl cursor-pointer mb-2 font-medium
        ${
          active
            ? "bg-purple-700 text-white"
            : "text-white/70 hover:bg-white/10 hover:text-white"
        }`}
    >
      {label}
    </div>
  );
}
