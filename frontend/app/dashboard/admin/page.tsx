import Sidebar from "@/app/components/Sidebar";

export default function AdminDashboard() {
  return (
    <div className="flex min-h-screen bg-gradient-to-br from-black via-[#0a1633] to-black">
      {/* Sidebar */}
      <Sidebar role="admin" />

      {/* Main */}
      <main className="flex-1 p-10">
        {/* Header */}
        <h1 className="text-3xl font-bold text-white mb-8">
          Admin Dashboard
        </h1>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
          <StatCard title="Total Exams" value="4" />
          <StatCard title="Students Appeared" value="96" />
          <StatCard title="Malpractice Cases" value="11" danger />
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          
          {/* AI SPECIAL REPORT (LEFT) */}
          <section
            className="
              xl:col-span-2
              bg-white/90 backdrop-blur-xl
              rounded-3xl shadow-2xl
              border border-black/10
              p-8
            "
          >
            {/* Top Accent */}
            <div className="h-1 w-full rounded-full bg-gradient-to-r from-[#5c145a] to-[#7a1c6b] mb-6" />

            <h2 className="text-2xl font-bold text-black mb-2">
              AI Special Report
            </h2>

            <p className="text-black/70 mb-6">
              High-risk events detected by the AI engine
            </p>

            {/* Alerts */}
            <div className="space-y-4 mb-8">
              <Alert text="Student 21 – Mobile phone detected" />
              <Alert text="Student 34 – Multiple faces detected" />
              <Alert text="Student 56 – Face not visible (15s)" />
            </div>

            {/* Actions */}
            <div className="flex gap-4">
              <button
                className="
                  px-6 py-3 rounded-xl
                  text-white font-semibold
                  bg-gradient-to-r from-[#5c145a] to-[#7a1c6b]
                  hover:brightness-110 transition
                  shadow-lg
                "
              >
                Review Exam
              </button>

              <button
                className="
                  px-6 py-3 rounded-xl
                  text-black font-semibold
                  bg-gray-200 hover:bg-gray-300 transition
                "
              >
                Download Report
              </button>
            </div>
          </section>

          {/* AI MONITORING STATUS (RIGHT) */}
          <section
            className="
              bg-white/90 backdrop-blur-xl
              rounded-3xl shadow-2xl
              border border-black/10
              p-8
            "
          >
            <h2 className="text-xl font-bold text-black mb-6">
              AI Monitoring Status
            </h2>

            <Status label="Face Detection" value="Running" />
            <Status label="Object Detection" value="Running" />
            <Status label="Audio Monitoring" value="Active" />
            <Status label="Malpractice Engine" value="Online" />

            <div className="mt-6 p-4 rounded-xl bg-green-50 border border-green-200">
              <p className="text-green-700 font-semibold">
                All AI systems operational
              </p>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}

/* ---------- COMPONENTS ---------- */

function StatCard({
  title,
  value,
  danger = false,
}: {
  title: string;
  value: string;
  danger?: boolean;
}) {
  return (
    <div className="bg-white/90 backdrop-blur-xl rounded-2xl shadow-xl p-6 border border-black/10">
      <p className="text-sm text-gray-500 mb-1">{title}</p>
      <p
        className={`text-3xl font-bold ${
          danger ? "text-red-600" : "text-black"
        }`}
      >
        {value}
      </p>
    </div>
  );
}

function Alert({ text }: { text: string }) {
  return (
    <div className="flex items-center gap-3 bg-red-50 border border-red-200 rounded-xl p-4">
      <span className="text-red-600 text-lg">⚠</span>
      <p className="text-red-700 font-medium">{text}</p>
    </div>
  );
}

function Status({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="flex justify-between items-center mb-4">
      <span className="text-black/70">{label}</span>
      <span className="text-green-600 font-semibold">{value}</span>
    </div>
  );
}
