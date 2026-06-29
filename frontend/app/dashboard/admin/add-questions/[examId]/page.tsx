"use client";

import { useParams } from "next/navigation";
import { useState } from "react";

const API_BASE = "http://localhost:5000";

export default function AddQuestionsPage() {
  const { examId } = useParams();

  const [questionText, setQuestionText] = useState("");
  const [options, setOptions] = useState(["", "", "", ""]);
  const [correctOptionIndex, setCorrectOptionIndex] = useState(0);
  const [loading, setLoading] = useState(false);

  const handleOptionChange = (index: number, value: string) => {
    const updated = [...options];
    updated[index] = value;
    setOptions(updated);
  };

  const handleAddQuestion = async () => {
    if (!questionText || options.some(o => !o)) {
      alert("Fill all fields");
      return;
    }

    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/questions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          examId, // ✅ THIS IS THE MOST IMPORTANT LINE
          questionText,
          options,
          correctOptionIndex,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Failed to add question");
      }

      // Reset form
      setQuestionText("");
      setOptions(["", "", "", ""]);
      setCorrectOptionIndex(0);

      alert("Question added ✅");
    } catch (err: any) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-10">
      <h1 className="text-3xl font-bold text-white mb-6">
        Add Questions
      </h1>

      <div className="bg-white rounded-2xl p-8 max-w-4xl shadow-xl">
        <input
          type="text"
          placeholder="Enter question"
          value={questionText}
          onChange={(e) => setQuestionText(e.target.value)}
          className="w-full p-3 mb-4 border rounded-lg"
        />

        {options.map((opt, i) => (
          <input
            key={i}
            type="text"
            placeholder={`Option ${i + 1}`}
            value={opt}
            onChange={(e) => handleOptionChange(i, e.target.value)}
            className="w-full p-3 mb-3 border rounded-lg"
          />
        ))}

        <select
          value={correctOptionIndex}
          onChange={(e) => setCorrectOptionIndex(Number(e.target.value))}
          className="p-3 mb-6 border rounded-lg"
        >
          <option value={0}>Option 1</option>
          <option value={1}>Option 2</option>
          <option value={2}>Option 3</option>
          <option value={3}>Option 4</option>
        </select>

        <button
          onClick={handleAddQuestion}
          disabled={loading}
          className="
            w-full py-4 rounded-xl
            text-white text-lg font-bold
            bg-gradient-to-r from-purple-700 to-pink-700
            hover:brightness-110
            disabled:opacity-50
          "
        >
          {loading ? "Saving..." : "➕ Add Question"}
        </button>
      </div>
    </div>
  );
}
