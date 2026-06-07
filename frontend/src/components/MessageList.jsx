import MessageBubble from "./MessageBubble.jsx";

export default function MessageList({ messages }) {
  if (!messages.length) {
    return (
      <div className="empty-state">
        <p className="eyebrow">Düzce Üniversitesi Öğrenci İşleri</p>
        <h2>Resmi kaynaklara dayalı soru-cevap asistanı</h2>
        <p>
          Staj, ders kaydı, sınavlar, yaz okulu ve belge işlemleri için konu
          başlığı seçebilir ya da doğrudan sorunuzu yazabilirsiniz.
        </p>
      </div>
    );
  }

  return (
    <div className="message-list">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
    </div>
  );
}
