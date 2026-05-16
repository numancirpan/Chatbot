const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function parseResponse(response) {
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const detail = data.detail || "API isteği başarısız oldu.";
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }

  return data;
}

export async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  return parseResponse(response);
}

export async function sendChatMessage({ query, sessionId, programScope }) {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      program_scope: programScope || null,
      session_id: sessionId,
    }),
  });

  return parseResponse(response);
}

export async function resetSessionMemory(sessionId) {
  const response = await fetch(`${API_BASE_URL}/reset-memory`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ session_id: sessionId }),
  });

  return parseResponse(response);
}
