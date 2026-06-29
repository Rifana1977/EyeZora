"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

const API_BASE = "http://localhost:5000";

export default function CreateTestPage() {
  const [title, setTitle] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleCreateExam = async () => {
    if (!title.trim()) {
      alert("Enter exam title");
      return;
    }

    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/admin/exam`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ title }),
      });

      // 🔥 SAFETY: read raw text first
      const rawText = await res.text();
      console.log("RAW RESPONSE FROM BACKEND:", rawText);

      let data;
      try {
        data = JSON.parse(rawText);
      } catch {
        throw new Error(
          "Backend did not return JSON. Check API route or server."
        );
      }

      if (!res.ok) {
        throw new Error(data.error || "Failed to create exam");
      }

      // ✅ redirect to add-questions page
      router.push(`/dashboard/admin/add-questions/${data._id}`);
    } catch (err: any) {
      console.error(err);
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-10">
      <h1 className="text-3xl font-bold text-white mb-8">
        Create Test
      </h1>

      <div className="bg-white rounded-2xl p-8 max-w-3xl shadow-xl">
        <label className="block text-gray-700 font-semibold mb-2">
          Exam Title
        </label>

        <input
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          className="
            w-full p-3 mb-6
            border border-gray-300 rounded-lg
            text-black
            focus:outline-none focus:ring-2 focus:ring-purple-600
          "
          placeholder="Enter exam title"
        />

        <button
          type="button"
          onClick={handleCreateExam}
          disabled={loading}
          className="
            w-full py-4 rounded-xl
            text-white text-lg font-semibold
            bg-gradient-to-r from-purple-700 to-pink-700
            hover:brightness-110
            transition
            shadow-lg
            disabled:opacity-50
            cursor-pointer
          "
        >
          {loading ? "Creating..." : "Create Exam"}
        </button>
      </div>
    </div>
  );
}


