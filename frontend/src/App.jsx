import ChatLayout from "./components/ChatLayout.jsx";
import { useChatSessions } from "./hooks/useChatSessions.js";
import { checkHealth, sendChatMessage } from "./services/chatApi.js";
import { useEffect, useState } from "react";

export default function App() {
  const chat = useChatSessions();
  const [health, setHealth] = useState({ status: "checking", message: "" });

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

  return <ChatLayout chat={chat} health={health} onSend={handleSend} />;
}
