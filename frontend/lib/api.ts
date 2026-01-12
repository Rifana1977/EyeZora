const BASE_URL = "http://localhost:5000/api";

export async function createQuestion(data: any) {
  const res = await fetch(`${BASE_URL}/questions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });

  if (!res.ok) {
    throw new Error("Failed to create question");
  }

  return res.json();
}

export async function getQuestionsByExam(examId: string) {
  const res = await fetch(`${BASE_URL}/questions/${examId}`);

  if (!res.ok) {
    throw new Error("Failed to fetch questions");
  }

  return res.json();
}
