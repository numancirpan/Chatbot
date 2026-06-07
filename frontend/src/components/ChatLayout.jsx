import { useState } from "react";
import ChatWindow from "./ChatWindow.jsx";
import Sidebar from "./Sidebar.jsx";

export default function ChatLayout({ chat, health, onSend, onClearSession }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="app-shell">
      <Sidebar
        sessions={chat.sessions}
        activeSessionId={chat.activeSessionId}
        onSelectSession={(id) => {
          chat.setActiveSessionId(id);
          setSidebarOpen(false);
        }}
        onNewSession={() => {
          chat.createNewSession();
          setSidebarOpen(false);
        }}
        onDeleteSession={chat.deleteSession}
        onClose={() => setSidebarOpen(false)}
        open={sidebarOpen}
      />
      {sidebarOpen ? (
        <button
          className="sidebar-overlay"
          type="button"
          onClick={() => setSidebarOpen(false)}
          aria-label="Sohbet menüsünü kapat"
        />
      ) : null}
      <ChatWindow
        session={chat.activeSession}
        health={health}
        programOptions={chat.programOptions}
        onSend={onSend}
        onClearSession={onClearSession}
        onProgramScopeChange={(programScope) =>
          chat.setSessionProgramScope(chat.activeSession.id, programScope)
        }
        onToggleSidebar={() => setSidebarOpen((open) => !open)}
      />
    </div>
  );
}
