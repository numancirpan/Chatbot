import { useEffect, useState } from "react";
import ChatLayout from "./components/ChatLayout.jsx";
import { useChatSessions } from "./hooks/useChatSessions.js";
import { checkHealth, resetSessionMemory, sendChatMessage } from "./services/chatApi.js";

export default function App() {
  const chat = useChatSessions();
  const [health, setHealth] = useState({ status: "checking", message: "" });
  const [toast, setToast] = useState(null);

  useEffect(() => {
    let isMounted = true;

    checkHealth()
      .then((data) => {
        if (isMounted) {
          setHealth({
            status: "ok",
            message: `${data.chunk_count ?? 0} chunk hazır`,
          });
        }
      })
      .catch(() => {
        if (isMounted) {
          setHealth({
            status: "error",
            message: "Backend bağlantısı bekleniyor",
          });
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    if (!toast) return undefined;

    const timeoutId = window.setTimeout(() => setToast(null), 2800);
    return () => window.clearTimeout(timeoutId);
  }, [toast]);

  const handleSend = async (text) => {
    const session = chat.activeSession;
    if (!session) return;

    chat.appendMessage(session.id, {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      sources: [],
      createdAt: new Date().toISOString(),
    });
    chat.setSessionLoading(session.id, true);
    chat.setSessionError(session.id, "");
    chat.renameSessionFromMessage(session.id, text);

    try {
      const response = await sendChatMessage({
        query: text,
        sessionId: session.id,
        programScope: session.programScope || null,
      });

      chat.appendMessage(session.id, {
        id: crypto.randomUUID(),
        role: "assistant",
        content: response.cevap,
        sources: response.kaynaklar || [],
        createdAt: new Date().toISOString(),
      });
    } catch (error) {
      chat.setSessionError(
        session.id,
        error.message || "Mesaj gönderilirken bir hata oluştu.",
      );
    } finally {
      chat.setSessionLoading(session.id, false);
    }
  };

  const handleClearSession = async () => {
    const session = chat.activeSession;
    if (!session) return;

    chat.clearSessionMessages(session.id);
    try {
      await resetSessionMemory(session.id);
      setToast({ type: "success", message: "Sohbet temizlendi." });
    } catch {
      const message = "Sohbet temizlendi, ancak backend hafızası sıfırlanamadı.";
      chat.setSessionError(session.id, message);
      setToast({ type: "warning", message });
    }
  };

  return (
    <>
      <ChatLayout
        chat={chat}
        health={health}
        onSend={handleSend}
        onClearSession={handleClearSession}
      />
      {toast ? (
        <div className={`toast ${toast.type}`} role="status">
          <span>{toast.message}</span>
          <button type="button" onClick={() => setToast(null)} aria-label="Bildirimi kapat">
            x
          </button>
        </div>
      ) : null}
    </>
  );
}
