export default function ChatHistory({ sessions, activeSessionId, onSelectSession }) {
  return (
    <nav className="chat-history" aria-label="Sohbet geçmişi">
      {sessions.map((session) => (
        <button
          className={`history-item ${session.id === activeSessionId ? "active" : ""}`}
          key={session.id}
          type="button"
          onClick={() => onSelectSession(session.id)}
        >
          <span className="history-title">{session.title}</span>
          <span className="history-meta">
            {session.messages.length ? `${session.messages.length} mesaj` : "Boş sohbet"}
          </span>
        </button>
      ))}
    </nav>
  );
}
