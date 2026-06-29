"use client";

import { useRouter } from "next/navigation";
import Sidebar from "@/app/components/Sidebar";

const EXAM_ID = "6967d63629e44c0adf1b5706"; 
// 👆 MUST be the SAME examId that admin used

export default function StudentDashboard() {
  const router = useRouter();

  return (
    <div className="flex min-h-screen bg-gradient-to-br from-black via-[#0a1633] to-black">
      {/* Sidebar */}
      <Sidebar role="student" />

      {/* Main Content */}
      <div className="flex-1 p-10">
        <h1 className="text-3xl font-bold text-white mb-8">
          Student Dashboard
        </h1>

        {/* Exam Card */}
        <div className="bg-white/90 backdrop-blur-xl rounded-3xl shadow-2xl p-8 max-w-4xl border border-black/10">
          <h2 className="text-2xl font-bold text-black mb-2">
            AI Proctored Examination
          </h2>

          <p className="text-black/70 mb-6">
            Please ensure your camera and microphone are enabled before starting.
          </p>

          {/* Info Grid */}
          <div className="grid grid-cols-2 gap-6 mb-8">
            <Info title="Duration" value="60 Minutes" />
            <Info title="Exam Type" value="AI Monitored" />
            <Info title="Camera Status" value="Ready" green />
            <Info title="Microphone" value="Ready" green />
          </div>

          {/* Instructions */}
          <div className="mb-8">
            <h3 className="font-semibold text-black mb-2">
              Important Instructions
            </h3>
            <ul className="text-black/70 list-disc ml-6 space-y-1">
              <li>Do not leave the camera view</li>
              <li>No mobile phones or additional persons</li>
              <li>Any malpractice will be AI-flagged</li>
            </ul>
          </div>

          {/* ✅ FIXED START BUTTON */}
          <button
            onClick={() =>
              router.push(`/dashboard/student/exam/${EXAM_ID}`)
            }
            className="
              w-full py-4 rounded-2xl
              text-white text-lg font-semibold
              bg-gradient-to-r from-[#5c145a] to-[#7a1c6b]
              hover:brightness-110 transition
              shadow-xl
            "
          >
            Start Test
          </button>
        </div>
      </div>
    </div>
  );
}

/* Info Card */
function Info({
  title,
  value,
  green = false,
}: {
  title: string;
  value: string;
  green?: boolean;
}) {
  return (
    <div className="bg-gray-100 rounded-xl p-4">
      <p className="text-sm text-gray-500">{title}</p>
      <p
        className={`text-lg font-semibold ${
          green ? "text-green-600" : "text-black"
        }`}
      >
        {value}
      </p>
    </div>
  );
}