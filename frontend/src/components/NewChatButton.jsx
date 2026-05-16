export default function NewChatButton({ onClick }) {
  return (
    <button className="new-chat-button" type="button" onClick={onClick}>
      <span aria-hidden="true">+</span>
      Yeni sohbet
    </button>
  );
}
