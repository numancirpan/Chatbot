import { useMemo, useState } from "react";

function formatDate(value) {
  return new Intl.DateTimeFormat("tr-TR", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function sessionMatches(session, query) {
  const normalized = query.trim().toLocaleLowerCase("tr-TR");
  if (!normalized) return true;

  const title = session.title.toLocaleLowerCase("tr-TR");
  const messages = session.messages
    .map((message) => message.content)
    .join(" ")
    .toLocaleLowerCase("tr-TR");
  return title.includes(normalized) || messages.includes(normalized);
}

export default function ChatHistory({
  sessions,
  activeSessionId,
  onSelectSession,
  onDeleteSession,
}) {
  const [query, setQuery] = useState("");
  const filteredSessions = useMemo(
    () => sessions.filter((session) => sessionMatches(session, query)),
    [query, sessions],
  );

  return (
    <div className="history-panel">
      <label className="history-search">
        <span>Sohbet ara</span>
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Başlık veya mesaj"
        />
      </label>
      <nav className="chat-history" aria-label="Sohbet geçmişi">
        {filteredSessions.map((session) => {
          const active = session.id === activeSessionId;

          return (
            <div className={`history-item ${active ? "active" : ""}`} key={session.id}>
              <button
                className="history-main"
                type="button"
                onClick={() => onSelectSession(session.id)}
                aria-current={active ? "page" : undefined}
              >
                <span className="history-title">{session.title}</span>
                <span className="history-meta">
                  {session.messages.length ? `${session.messages.length} mesaj` : "Boş sohbet"}
                  {" · "}
                  {formatDate(session.updatedAt)}
                </span>
              </button>
              <button
                className="history-delete"
                type="button"
                onClick={() => onDeleteSession(session.id)}
                aria-label={`${session.title} sohbetini sil`}
                title="Sohbeti sil"
              >
                x
              </button>
            </div>
          );
        })}
        {!filteredSessions.length ? (
          <div className="history-empty">Aramanızla eşleşen sohbet yok.</div>
        ) : null}
      </nav>
    </div>
  );
}
