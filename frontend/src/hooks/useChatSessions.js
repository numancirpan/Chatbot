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

const PROGRAM_OPTIONS = [
  { value: "", label: "Genel" },
  { value: "bilgisayar_muhendisligi", label: "Bilgisayar Mühendisliği" },
  { value: "orman_muhendisligi", label: "Orman Mühendisliği" },
  { value: "insaat_muhendisligi", label: "İnşaat Mühendisliği" },
  { value: "mimarlik", label: "Mimarlık" },
  { value: "isletme", label: "İşletme" },
];

function isBlankSession(session) {
  return (
    session?.title === "Yeni sohbet" &&
    !session?.programScope &&
    !session?.messages?.length
  );
}

function readSessions() {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    if (Array.isArray(stored)) {
      return stored.filter((session) => !isBlankSession(session));
    }
  } catch {
    localStorage.removeItem(STORAGE_KEY);
  }

  return [];
}

function initializeSessions() {
  return [createSession(), ...readSessions()];
}

function persistSessions(nextSessions) {
  const serializable = nextSessions
    .filter((session) => !isBlankSession(session))
    .map(({ loading, error, ...session }) => ({
      ...session,
      loading: false,
      error: error || "",
    }));
  localStorage.setItem(STORAGE_KEY, JSON.stringify(serializable));
}

export function useChatSessions() {
  const [sessions, setSessions] = useState(initializeSessions);
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
    updateSessions((current) => [session, ...current.filter((item) => !isBlankSession(item))]);
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

  const setSessionProgramScope = (sessionId, programScope) => {
    updateSessions((current) =>
      current.map((session) =>
        session.id === sessionId
          ? { ...session, programScope, updatedAt: new Date().toISOString() }
          : session,
      ),
    );
  };

  const clearSessionMessages = (sessionId) => {
    updateSessions((current) =>
      current.map((session) =>
        session.id === sessionId
          ? {
              ...session,
              title: "Yeni sohbet",
              messages: [],
              loading: false,
              error: "",
              updatedAt: new Date().toISOString(),
            }
          : session,
      ),
    );
  };

  const deleteSession = (sessionId) => {
    updateSessions((current) => {
      const next = current.filter((session) => session.id !== sessionId);
      return next.length ? next : [createSession()];
    });
    setActiveSessionId((currentId) => {
      if (currentId !== sessionId) return currentId;
      const remaining = sessions.filter((session) => session.id !== sessionId);
      return remaining[0]?.id;
    });
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
    setSessionProgramScope,
    clearSessionMessages,
    deleteSession,
    programOptions: PROGRAM_OPTIONS,
  };
}
