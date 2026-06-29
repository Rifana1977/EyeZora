"use client";

import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

const API_BASE = "http://localhost:5000";

type Question = {
  _id: string;
  questionText: string;
  options: string[];
};

export default function StudentExamPage() {
  const { examId } = useParams();
  const [questions, setQuestions] = useState<Question[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchQuestions = async () => {
      console.log("🎯 Fetching questions for examId:", examId);

      const res = await fetch(
        `${API_BASE}/api/student/exam/${examId}`
      );

      const data = await res.json();
      console.log("📦 Questions received:", data);

      setQuestions(data);
      setLoading(false);
    };

    fetchQuestions();
  }, [examId]);

  if (loading) {
    return <p className="text-white p-10">Loading questions...</p>;
  }

  return (
    <div className="min-h-screen p-10 bg-black text-white">
      <h1 className="text-3xl font-bold mb-8">Exam in Progress</h1>

      {questions.length === 0 && (
        <p className="text-red-400">
          ❌ No questions found for this exam
        </p>
      )}

      {questions.map((q, i) => (
        <div
          key={q._id}
          className="bg-white text-black p-6 rounded-xl mb-6"
        >
          <p className="font-semibold mb-4">
            {i + 1}. {q.questionText}
          </p>

          {q.options.map((opt, idx) => (
            <div key={idx} className="mb-2">
              <label>
                <input type="radio" name={q._id} /> {opt}
              </label>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}