"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

export default function ExamPage() {
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);

  const STUDENT_ID = "STUDENT_21";
  const EXAM_ID = "EXAM01";

  const [permissionGranted, setPermissionGranted] = useState(false);
  const [timeLeft, setTimeLeft] = useState(60 * 60); // 1 hour

  const [faceLostSince, setFaceLostSince] = useState<number | null>(null);
  const [cameraLostSince, setCameraLostSince] = useState<number | null>(null);

  const [faceViolated, setFaceViolated] = useState(false);
  const [cameraViolated, setCameraViolated] = useState(false);

  /* 🔗 LOG EVENT (Node / Mongo) */
  const logEvent = async (event: string) => {
    try {
      await fetch("http://127.0.0.1:8000/verify-person", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          student_id: STUDENT_ID,
          exam_id: EXAM_ID,
          ai_result: event,
        }),
      });
    } catch (err) {
      console.error("Logging failed", err);
    }
  };

  /* 🔒 BLOCK BACK BUTTON */
  useEffect(() => {
    history.pushState(null, "", location.href);
    window.onpopstate = () => history.go(1);
  }, []);

  /* ⏱ EXAM TIMER */
  useEffect(() => {
    if (!permissionGranted) return;

    const timer = setInterval(() => {
      setTimeLeft((t) => {
        if (t <= 1) {
          clearInterval(timer);
          alert("Exam completed");
          router.push("/dashboard/student");
          return 0;
        }
        return t - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [permissionGranted, router]);

  /* 🎥 START CAMERA */
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      setPermissionGranted(true);
      logEvent("CAMERA_GRANTED");
    } catch {
      alert("Camera & microphone permission is mandatory");
    }
  };

  /* 🧠 AI ANALYSIS (best.pt) */
  const analyzeFrameWithAI = async () => {
    if (!videoRef.current) return null;

    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    ctx.drawImage(videoRef.current, 0, 0);
    const image = canvas.toDataURL("image/jpeg");

    const res = await fetch("http://127.0.0.1:8000/analyze-frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image,
        student_id: STUDENT_ID,
        exam_id: EXAM_ID,
      }),
    });

    const data = await res.json();
    return data.ai_result; // MATCH | NO_FACE | MULTIPLE_FACES
  };

  /* 👁 AI PROCTORING LOOP */
  useEffect(() => {
    if (!permissionGranted || !videoRef.current) return;

    const interval = setInterval(async () => {
      const video = videoRef.current;

      /* CAMERA CHECK */
      if (!video || video.readyState < 2) {
        if (!cameraLostSince) {
          setCameraLostSince(Date.now());
        } else if (
          Date.now() - cameraLostSince > 5000 &&
          !cameraViolated
        ) {
          alert("⚠ Camera disconnected!");
          logEvent("CAMERA_DISCONNECTED");
          setCameraViolated(true);
        }
        return;
      } else {
        setCameraLostSince(null);
        setCameraViolated(false);
      }

      /* AI FACE CHECK */
      const aiResult = await analyzeFrameWithAI();

      if (aiResult === "NO_FACE") {
        if (!faceLostSince) {
          setFaceLostSince(Date.now());
        } else if (
          Date.now() - faceLostSince > 5000 &&
          !faceViolated
        ) {
          alert("⚠ Face not visible!");
          logEvent("NO_FACE");
          setFaceViolated(true);
        }
      } else if (aiResult === "MULTIPLE_FACES") {
        alert("🚨 Multiple faces detected!");
        logEvent("MULTIPLE_FACES");
      } else {
        setFaceLostSince(null);
        setFaceViolated(false);
      }
    }, 5000); // every 5 sec

    return () => clearInterval(interval);
  }, [
    permissionGranted,
    faceLostSince,
    cameraLostSince,
    faceViolated,
    cameraViolated,
  ]);

  const formatTime = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  return (
    <div className="min-h-screen bg-black text-white p-6">
      <div className="flex justify-between mb-6">
        <h1 className="text-xl font-bold">AI Proctored Examination</h1>
        <div>⏱ {formatTime(timeLeft)}</div>
      </div>

      {!permissionGranted ? (
        <div className="max-w-xl mx-auto bg-white text-black p-6 rounded-xl">
          <h2 className="text-xl font-bold mb-4">Exam Rules</h2>
          <ul className="list-disc ml-5 mb-6">
            <li>Camera must stay ON</li>
            <li>Only one face allowed</li>
            <li>No leaving the screen</li>
          </ul>

          <button
            onClick={startCamera}
            className="w-full bg-purple-700 text-white py-2 rounded"
          >
            Allow Camera & Microphone
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-4 gap-6">
          <div className="col-span-3 bg-white text-black p-6 rounded-xl">
            <h2 className="font-bold mb-4">Question 1</h2>
            <p>What is the purpose of AI proctoring?</p>
          </div>

          <div className="bg-gray-800 p-4 rounded-xl text-center">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="w-full h-40 rounded"
            />
            <p className="mt-2 text-green-400">Camera Active</p>
          </div>
        </div>
      )}
    </div>
  );
}
