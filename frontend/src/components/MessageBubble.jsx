function SourceList({ sources }) {
  const uniqueSources = [];
  const seen = new Set();

  for (const source of sources || []) {
    const key = source.url || `${source.baslik}-${source.kategori}`;
    if (!seen.has(key)) {
      seen.add(key);
      uniqueSources.push(source);
    }
  }

  if (!uniqueSources.length) return null;

  return (
    <div className="sources">
      <span>Kaynaklar</span>
      {uniqueSources.map((source, index) => {
        const label = source.baslik || source.kategori || `Kaynak ${index + 1}`;
        return source.url?.startsWith("http") ? (
          <a key={`${source.url}-${index}`} href={source.url} target="_blank" rel="noreferrer">
            {label}
          </a>
        ) : (
          <p key={`${label}-${index}`}>{label}</p>
        );
      })}
    </div>
  );
}

export default function MessageBubble({ message }) {
  const isUser = message.role === "user";

  return (
    <article className={`message-row ${isUser ? "user" : "assistant"}`}>
      <div className="message-bubble">
        <div className="message-author">{isUser ? "Siz" : "Asistan"}</div>
        <p>{message.content}</p>
        <SourceList sources={message.sources} />
      </div>
    </article>
  );
}
