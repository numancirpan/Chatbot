import MessageBubble from "./MessageBubble.jsx";

export default function MessageList({ messages }) {
  if (!messages.length) {
    return (
      <div className="empty-state">
        <h2>Merhaba, nasıl yardımcı olabilirim?</h2>
        <p>Staj, ders kaydı, sınavlar, yaz okulu veya öğrenci işleri süreçleri hakkında soru sorabilirsiniz.</p>
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
