"use client";

export default function Sidebar({
  role,
}: {
  role: "student" | "admin";
}) {
  return (
    <aside
      className="
        w-72 min-h-screen
        bg-gradient-to-b from-black via-[#0a1633] to-black
        border-r border-white/10
        px-6 py-8
      "
    >
      {/* Brand */}
      <div className="mb-12">
        <h1 className="text-2xl font-extrabold text-white tracking-wide">
          EYEZORA
        </h1>
      </div>

      {/* Navigation */}
      <nav className="space-y-3">
        {role === "student" ? (
          <>
            <Item label="My Exam" active />
            <Item label="Instructions" />
            <Item label="Result" />
          </>
        ) : (
          <>
            <Item label="Dashboard" active />
            <Item label="Create Test" />
            <Item label="Reports" />
          </>
        )}
      </nav>
    </aside>
  );
}

function Item({
  label,
  active = false,
}: {
  label: string;
  active?: boolean;
}) {
  return (
    <div
      className={`
        px-4 py-3 rounded-xl cursor-pointer
        font-medium transition-all
        ${
          active
            ? "bg-gradient-to-r from-[#5c145a] to-[#7a1c6b] text-white shadow-lg"
            : "text-white/70 hover:text-white hover:bg-white/10"
        }
      `}
    >
      {label}
    </div>
  );
}