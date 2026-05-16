import ErrorState from "./ErrorState.jsx";
import LoadingState from "./LoadingState.jsx";
import MessageInput from "./MessageInput.jsx";
import MessageList from "./MessageList.jsx";

export default function ChatWindow({ session, health, onSend, onToggleSidebar }) {
  return (
    <main className="chat-window">
      <header className="topbar">
        <button
          className="icon-button mobile-menu"
          type="button"
          onClick={onToggleSidebar}
          aria-label="Sohbet listesini aç"
          title="Sohbet listesi"
        >
          ☰
        </button>
        <div>
          <h1>{session?.title || "Yeni sohbet"}</h1>
          <p className={`health ${health.status}`}>{health.message}</p>
        </div>
      </header>

      <section className="conversation" aria-live="polite">
        <MessageList messages={session?.messages || []} />
        {session?.loading ? <LoadingState /> : null}
        {session?.error ? <ErrorState message={session.error} /> : null}
      </section>

      <MessageInput disabled={session?.loading} onSend={onSend} />
    </main>
  );
}
