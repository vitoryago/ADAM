const API_BASE = "/api/deep-discussion";

async function throwIfResNotOk(res: Response) {
  if (!res.ok) {
    const text = (await res.text()) || res.statusText;
    throw new Error(`${res.status}: ${text}`);
  }
}

export async function createSession(
  projectId: string,
  question: string,
  pattern: string,
  conversationId?: string,
  preferLocal?: boolean,
) {
  const res = await fetch(`${API_BASE}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      project_id: projectId,
      question,
      pattern,
      conversation_id: conversationId,
      prefer_local: preferLocal ?? false,
    }),
    credentials: "include",
  });
  await throwIfResNotOk(res);
  return res.json();
}

export async function updateSessionConfig(
  sessionId: string,
  config: {
    model_assignments?: Record<string, string>;
    pattern?: string;
    budget?: number;
  },
) {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/config`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
    credentials: "include",
  });
  await throwIfResNotOk(res);
  return res.json();
}

export function startSession(sessionId: string): EventSource {
  return new EventSource(`${API_BASE}/sessions/${sessionId}/run`);
}

export async function getSession(sessionId: string) {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}`, {
    credentials: "include",
  });
  await throwIfResNotOk(res);
  return res.json();
}

export async function listSessions(projectId: string) {
  const res = await fetch(`${API_BASE}/sessions?project_id=${projectId}`, {
    credentials: "include",
  });
  await throwIfResNotOk(res);
  return res.json();
}

export async function replaySession(sessionId: string) {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/replay`, {
    method: "POST",
    credentials: "include",
  });
  await throwIfResNotOk(res);
  return res.json();
}

export async function createFromConversation(
  conversationId: string,
  question: string,
) {
  const res = await fetch(`${API_BASE}/from-conversation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversation_id: conversationId, question }),
    credentials: "include",
  });
  await throwIfResNotOk(res);
  return res.json();
}
