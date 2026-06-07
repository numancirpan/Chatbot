import { useState } from "react";

const NO_ANSWER_TERMS = [
  "kesin cevap veremiyorum",
  "net cevap veremiyorum",
  "kaynak bulamadım",
  "resmi kaynak bulamadım",
  "yanlış yönlendirmemek",
];

function normalizeText(value) {
  return value.toLocaleLowerCase("tr-TR");
}

function isNoAnswerMessage(message) {
  if (message.role !== "assistant") return false;
  const content = normalizeText(message.content || "");
  return NO_ANSWER_TERMS.some((term) => content.includes(term));
}

function confidenceMeta(source) {
  const rawScore = Number(source.guven_skoru ?? source.confidence ?? source.score);
  if (!Number.isFinite(rawScore)) {
    return { label: "Güven bilgisi yok", tone: "neutral" };
  }

  if (rawScore >= 75) return { label: "Güven yüksek", tone: "high" };
  if (rawScore >= 45) return { label: "Güven orta", tone: "medium" };
  return { label: "Güven düşük", tone: "low" };
}

function SourceList({ sources }) {
  const [open, setOpen] = useState(true);
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
    <div className={`sources ${open ? "sources-open" : ""}`}>
      <button
        className="sources-header"
        type="button"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={open}
      >
        <span>Resmi kaynaklar</span>
        <small>{uniqueSources.length}</small>
        <span className="sources-caret">⌄</span>
      </button>
      {open ? (
        <div className="source-list">
          {uniqueSources.map((source, index) => {
            const label = source.baslik || source.kategori || `Kaynak ${index + 1}`;
            const meta = source.kategori || "Resmi kaynak";
            const confidence = confidenceMeta(source);
            const content = (
              <>
                <span className="source-badge">Resmi</span>
                <span className="source-title">{label}</span>
                <span className="source-meta">{meta}</span>
                <span className={`source-confidence ${confidence.tone}`}>
                  {confidence.label}
                </span>
                {source.url?.startsWith("http") ? <span className="source-open">Aç ↗</span> : null}
              </>
            );

            return source.url?.startsWith("http") ? (
              <a
                className="source-item"
                key={`${source.url}-${index}`}
                href={source.url}
                target="_blank"
                rel="noreferrer"
              >
                {content}
              </a>
            ) : (
              <div className="source-item" key={`${label}-${index}`}>
                {content}
              </div>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

function NoAnswerNotice() {
  return (
    <div className="no-answer-notice" role="note">
      <strong>Yeterli resmi kanıt bulunamadı</strong>
      <span>
        Bu cevapta model tahmin yürütmek yerine kaynak eksikliğini bildiriyor. Daha net sonuç için
        soruyu program, dönem veya işlem adıyla daraltabilirsiniz.
      </span>
    </div>
  );
}

function formatTime(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat("tr-TR", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

export default function MessageBubble({ message }) {
  const isUser = message.role === "user";
  const noAnswer = isNoAnswerMessage(message);

  return (
    <article className={`message-row ${isUser ? "user" : "assistant"}`}>
      <div className={`message-bubble ${noAnswer ? "no-answer" : ""}`}>
        <div className="message-author">
          <span>{isUser ? "Siz" : "Asistan"}</span>
          <time>{formatTime(message.createdAt)}</time>
        </div>
        {noAnswer ? <NoAnswerNotice /> : null}
        <p>{message.content}</p>
        <SourceList sources={message.sources} />
      </div>
    </article>
  );
}
