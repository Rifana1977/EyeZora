"use client";

import { useEffect, useState } from "react";

/* ---------------- TYPES ---------------- */

type BackendOption =
  | string
  | {
      text: string;
    };

type Question = {
  _id?: string;
  examId: string;
  questionText: string;
  options: BackendOption[];
  correctOptionIndex: number;
};

/* ---------------- CONFIG ---------------- */

const EXAM_ID = "695d2959c07b91e4b64aece1"; // ✅ your real examId
const API_BASE = "http://localhost:5000/api/questions";

/* ---------------- COMPONENT ---------------- */

export default function AdminQuestionsPage() {
  const [questionText, setQuestionText] = useState("");
  const [options, setOptions] = useState<string[]>(["", "", "", ""]);
  const [correctIndex, setCorrectIndex] = useState(0);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [loading, setLoading] = useState(false);

  /* ---------------- FETCH QUESTIONS ---------------- */

  useEffect(() => {
    fetchQuestions();
  }, []);

  const fetchQuestions = async () => {
    try {
      const res = await fetch(`${API_BASE}/${EXAM_ID}`);
      const data = await res.json();
      setQuestions(data);
    } catch (err) {
      console.error("Failed to fetch questions", err);
    }
  };

  /* ---------------- SAVE QUESTION ---------------- */

  const handleSubmit = async () => {
    if (!questionText || options.some(o => !o)) {
      alert("Fill all fields");
      return;
    }

    setLoading(true);

    try {
      const res = await fetch(API_BASE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          examId: EXAM_ID,
          questionText,
          options: options.map(o => ({ text: o })), // ✅ backend-safe
          correctOptionIndex: correctIndex,
        }),
      });

      const savedQuestion = await res.json();

      // ✅ instantly show new question
      setQuestions(prev => [...prev, savedQuestion]);

      // reset form
      setQuestionText("");
      setOptions(["", "", "", ""]);
      setCorrectIndex(0);
    } catch (err) {
      console.error("Error saving question", err);
    } finally {
      setLoading(false);
    }
  };

  /* ---------------- UI ---------------- */

  return (
    <div className="p-10">
      <h1 className="text-2xl font-bold mb-6">Create Test</h1>

      {/* Question */}
      <input
        className="border p-2 w-full mb-3"
        placeholder="Question"
        value={questionText}
        onChange={e => setQuestionText(e.target.value)}
      />

      {/* Options */}
      {options.map((opt, i) => (
        <input
          key={i}
          className="border p-2 w-full mb-2"
          placeholder={`Option ${i + 1}`}
          value={opt}
          onChange={e => {
            const copy = [...options];
            copy[i] = e.target.value;
            setOptions(copy);
          }}
        />
      ))}

      {/* Correct option */}
      <select
        className="border p-2 mb-4"
        value={correctIndex}
        onChange={e => setCorrectIndex(Number(e.target.value))}
      >
        {options.map((_, i) => (
          <option key={i} value={i}>
            Correct Option {i + 1}
          </option>
        ))}
      </select>

      <br />

      <button
        onClick={handleSubmit}
        disabled={loading}
        className="bg-purple-600 text-white px-6 py-2 rounded disabled:opacity-50"
      >
        {loading ? "Saving..." : "Save Question"}
      </button>

      <hr className="my-10" />

      {/* Questions List */}
      <h2 className="text-xl font-semibold mb-4">Questions List</h2>

      {questions.map((q, i) => (
        <div key={q._id || i} className="mb-4 border p-4 rounded">
          <p className="font-medium">
            {i + 1}. {q.questionText}
          </p>

          <ul className="ml-4 list-disc">
            {q.options.map((opt, idx) => {
              const text =
                typeof opt === "string" ? opt : opt.text;

              return (
                <li
                  key={idx}
                  className={
                    idx === q.correctOptionIndex
                      ? "text-green-600 font-semibold"
                      : ""
                  }
                >
                  {text}
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </div>
  );
}
