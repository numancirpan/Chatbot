import { useState } from "react";

export default function MessageInput({ disabled, onSend }) {
  const [value, setValue] = useState("");

  const submit = (event) => {
    event.preventDefault();
    const text = value.trim();
    if (!text || disabled) return;

    onSend(text);
    setValue("");
  };

  return (
    <form className="message-input" onSubmit={submit}>
      <textarea
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            submit(event);
          }
        }}
        placeholder="Sorunuzu yazın..."
        rows={1}
        disabled={disabled}
      />
      <button type="submit" disabled={disabled || !value.trim()} title="Gönder">
        Gönder
      </button>
    </form>
  );
}
