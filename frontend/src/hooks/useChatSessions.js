import { useMemo, useState } from "react";

const STORAGE_KEY = "duzce-rag-chat-sessions";

function createId() {
  if (crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `chat-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function createSession() {
  const now = new Date().toISOString();
  return {
    id: createId(),
    title: "Yeni sohbet",
    messages: [],
    programScope: "",
    loading: false,
    error: "",
    createdAt: now,
    updatedAt: now,
  };
}

function readSessions() {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    if (Array.isArray(stored) && stored.length > 0) {
      return stored;
    }
  } catch {
    localStorage.removeItem(STORAGE_KEY);
  }

  return [createSession()];
}

function persistSessions(nextSessions) {
  const serializable = nextSessions.map(({ loading, error, ...session }) => ({
    ...session,
    loading: false,
    error: error || "",
  }));
  localStorage.setItem(STORAGE_KEY, JSON.stringify(serializable));
}

export function useChatSessions() {
  const [sessions, setSessions] = useState(readSessions);
  const [activeSessionId, setActiveSessionId] = useState(() => sessions[0]?.id);
  const activeSession = useMemo(
    () => sessions.find((session) => session.id === activeSessionId) || sessions[0],
    [activeSessionId, sessions],
  );

  const updateSessions = (updater) => {
    setSessions((current) => {
      const next = updater(current);
      persistSessions(next);
      return next;
    });
  };

  const createNewSession = () => {
    const session = createSession();
    updateSessions((current) => [session, ...current]);
    setActiveSessionId(session.id);
  };

  const appendMessage = (sessionId, message) => {
    updateSessions((current) =>
      current.map((session) =>
        session.id === sessionId
          ? {
              ...session,
              messages: [...session.messages, message],
              updatedAt: new Date().toISOString(),
            }
          : session,
      ),
    );
  };

  const renameSessionFromMessage = (sessionId, message) => {
    const cleanTitle = message.trim().replace(/\s+/g, " ").slice(0, 52);
    if (!cleanTitle) return;

    updateSessions((current) =>
      current.map((session) =>
        session.id === sessionId && session.title === "Yeni sohbet"
          ? { ...session, title: cleanTitle, updatedAt: new Date().toISOString() }
          : session,
      ),
    );
  };

  const setSessionLoading = (sessionId, loading) => {
    updateSessions((current) =>
      current.map((session) =>
        session.id === sessionId ? { ...session, loading } : session,
      ),
    );
  };

  const setSessionError = (sessionId, error) => {
    updateSessions((current) =>
      current.map((session) =>
        session.id === sessionId ? { ...session, error } : session,
      ),
    );
  };

  return {
    sessions,
    activeSession,
    activeSessionId,
    setActiveSessionId,
    createNewSession,
    appendMessage,
    renameSessionFromMessage,
    setSessionLoading,
    setSessionError,
  };
}
