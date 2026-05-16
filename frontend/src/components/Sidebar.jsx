import ChatHistory from "./ChatHistory.jsx";
import NewChatButton from "./NewChatButton.jsx";

export default function Sidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  open,
}) {
  return (
    <aside className={`sidebar ${open ? "sidebar-open" : ""}`}>
      <div className="brand">
        <div className="brand-mark">DÜ</div>
        <div>
          <p>Düzce Üniversitesi</p>
          <span>Öğrenci İşleri Asistanı</span>
        </div>
      </div>
      <NewChatButton onClick={onNewSession} />
      <ChatHistory
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={onSelectSession}
      />
    </aside>
  );
}
