import { useState } from "react";
import ErrorState from "./ErrorState.jsx";
import LoadingState from "./LoadingState.jsx";
import MessageInput from "./MessageInput.jsx";
import MessageList from "./MessageList.jsx";

const SUGGESTED_QUESTIONS = [
  {
    category: "Staj",
    items: [
      "Zorunlu staj kaç iş günü?",
      "Staj defteri ne zaman teslim edilir?",
      "Staj için hangi belgeler gerekir?",
    ],
  },
  {
    category: "Ders Kaydı",
    items: [
      "Ders kaydında danışman onayı gerekiyor mu?",
      "Kayıt yenileme ne zaman tamamlanır?",
      "Üstten ders alma şartı nedir?",
    ],
  },
  {
    category: "Yaz Okulu",
    items: [
      "Yaz okulunda en fazla kaç AKTS alınabilir?",
      "Yaz okulu dersleri nasıl onaylanır?",
      "Başka üniversiteden yaz okulu alınabilir mi?",
    ],
  },
  {
    category: "Belgeler",
    items: [
      "Transkript nereden alınır?",
      "Öğrenci belgesi nasıl alınır?",
      "Askerlik tecil belgesi için ne gerekir?",
    ],
  },
];

export default function ChatWindow({
  session,
  health,
  programOptions,
  onSend,
  onClearSession,
  onProgramScopeChange,
  onToggleSidebar,
}) {
  const [confirmClear, setConfirmClear] = useState(false);
  const hasMessages = Boolean(session?.messages?.length);

  const handleClear = () => {
    setConfirmClear(false);
    onClearSession();
  };

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
        <div className="topbar-title">
          <h1>{session?.title || "Yeni sohbet"}</h1>
          <p className={`health ${health.status}`}>{health.message}</p>
        </div>
        <div className="topbar-controls" aria-label="Sohbet kontrolleri">
          <label className="scope-select">
            <span>Program</span>
            <select
              value={session?.programScope || ""}
              onChange={(event) => onProgramScopeChange(event.target.value)}
              disabled={session?.loading}
            >
              {programOptions.map((option) => (
                <option key={option.value || "general"} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <div className="clear-control">
            <button
              className="clear-button"
              type="button"
              onClick={() => setConfirmClear((current) => !current)}
              disabled={session?.loading || !hasMessages}
              aria-expanded={confirmClear}
            >
              Temizle
            </button>
            {confirmClear ? (
              <div className="clear-confirm" role="dialog" aria-label="Sohbeti temizle">
                <span>Bu sohbet temizlensin mi?</span>
                <div>
                  <button
                    type="button"
                    className="confirm-cancel"
                    onClick={() => setConfirmClear(false)}
                  >
                    Vazgeç
                  </button>
                  <button type="button" className="confirm-danger" onClick={handleClear}>
                    Temizle
                  </button>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </header>

      <section className="conversation" aria-live="polite">
        <MessageList messages={session?.messages || []} />
        {!hasMessages ? (
          <div className="prompt-groups" aria-label="Örnek sorular">
            {SUGGESTED_QUESTIONS.map((group) => (
              <section className="prompt-group" key={group.category}>
                <h3>{group.category}</h3>
                <div className="prompt-grid">
                  {group.items.map((question) => (
                    <button
                      key={question}
                      type="button"
                      onClick={() => onSend(question)}
                      disabled={session?.loading}
                    >
                      <strong>{question}</strong>
                    </button>
                  ))}
                </div>
              </section>
            ))}
          </div>
        ) : null}
        {session?.loading ? <LoadingState /> : null}
        {session?.error ? <ErrorState message={session.error} /> : null}
      </section>

      <MessageInput disabled={session?.loading} onSend={onSend} />
    </main>
  );
}
