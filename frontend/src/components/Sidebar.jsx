import ChatHistory from "./ChatHistory.jsx";
import NewChatButton from "./NewChatButton.jsx";
import duzceLogo from "../assets/duzce-university-logo.png";

export default function Sidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  onClose,
  open,
}) {
  return (
    <aside className={`sidebar ${open ? "sidebar-open" : ""}`}>
      <div className="sidebar-head">
        <div className="brand">
          <img className="brand-logo" src={duzceLogo} alt="Düzce Üniversitesi" />
          <div className="brand-copy">
            <p>Düzce Üniversitesi</p>
            <span>Öğrenci İşleri Asistanı</span>
          </div>
        </div>
        <button
          className="sidebar-close"
          type="button"
          onClick={onClose}
          aria-label="Geçmiş sohbet menüsünü küçült"
          title="Menüyü küçült"
        >
          ‹
        </button>
      </div>
      <NewChatButton onClick={onNewSession} />
      <ChatHistory
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={onSelectSession}
        onDeleteSession={onDeleteSession}
      />
    </aside>
  );
}
