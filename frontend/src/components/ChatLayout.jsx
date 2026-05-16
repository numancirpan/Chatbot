import { useState } from "react";
import Sidebar from "./Sidebar.jsx";
import ChatWindow from "./ChatWindow.jsx";

export default function ChatLayout({ chat, health, onSend }) {
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
        open={sidebarOpen}
      />
      <ChatWindow
        session={chat.activeSession}
        health={health}
        onSend={onSend}
        onToggleSidebar={() => setSidebarOpen((open) => !open)}
      />
    </div>
  );
}
