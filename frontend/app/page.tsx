"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

export default function LoginPage() {
  const router = useRouter();

  const [role, setRole] = useState<"student" | "admin">("student");
  const [open, setOpen] = useState(false);

  const login = () => {
    role === "admin"
      ? router.push("/dashboard/admin")
      : router.push("/dashboard/student");
  };

  return (
    <div
      className="
        min-h-screen flex flex-col items-center justify-center
        bg-gradient-to-br from-black via-[#0a1633] to-black
      "
    >
      {/* BRAND */}
      <h1 className="text-5xl font-extrabold tracking-widest text-white mb-10">
        EYEZORA
      </h1>

      {/* LOGIN CARD */}
      <div
        className="
          bg-[#ededf0]
          rounded-3xl p-10 w-[420px]
          shadow-2xl
        "
      >
        {/* TITLE */}
        <h2 className="text-3xl font-extrabold text-black text-center mb-1">
          AI Exam Proctoring
        </h2>

        <p className="text-black/70 text-center mb-8">
          Secure • Monitored • Intelligent
        </p>

        {/* USER ID */}
        <input
          placeholder="User ID"
          className="
            w-full mb-4 p-3 rounded-xl
            bg-white text-black
            border border-black/20
            focus:outline-none focus:ring-2 focus:ring-[#7a1c6b]
          "
        />

        {/* PASSWORD */}
        <input
          type="password"
          placeholder="Password"
          className="
            w-full mb-4 p-3 rounded-xl
            bg-white text-black
            border border-black/20
            focus:outline-none focus:ring-2 focus:ring-[#7a1c6b]
          "
        />

        {/* ROLE DROPDOWN */}
        <div className="relative mb-6">
          <button
            type="button"
            onClick={() => setOpen(!open)}
            className="
              w-full p-3 rounded-xl
              bg-white text-black font-semibold
              border-2 border-[#7a1c6b]
              flex justify-between items-center
              focus:outline-none focus:ring-2 focus:ring-[#7a1c6b]/40
            "
          >
            <span className="capitalize">{role}</span>
            <span className="text-black/70">▼</span>
          </button>

          {open && (
            <div
              className="
                absolute z-30 mt-2 w-full
                bg-white rounded-xl
                shadow-[0_15px_40px_rgba(0,0,0,0.25)]
                border border-black/10
                overflow-hidden
              "
            >
              <div
                onClick={() => {
                  setRole("student");
                  setOpen(false);
                }}
                className="
                  px-4 py-3 cursor-pointer
                  text-black font-medium
                  hover:bg-[#f3e8f3]
                "
              >
                Student
              </div>

              <div
                onClick={() => {
                  setRole("admin");
                  setOpen(false);
                }}
                className="
                  px-4 py-3 cursor-pointer
                  text-black font-medium
                  hover:bg-[#f3e8f3]
                "
              >
                Admin
              </div>
            </div>
          )}
        </div>

        {/* LOGIN BUTTON */}
        <button
          onClick={login}
          className="
            w-full py-3 rounded-xl
            text-white text-lg font-semibold
            bg-gradient-to-r from-[#5c145a] to-[#7a1c6b]
            hover:brightness-110 transition
            shadow-lg
          "
        >
          Login
        </button>
      </div>
    </div>
  );
}
