import { useRef, useState } from "react";

export default function MessageInput({ disabled, onSend }) {
  const [value, setValue] = useState("");
  const textareaRef = useRef(null);

  const resetTextarea = () => {
    requestAnimationFrame(() => {
      if (textareaRef.current) {
        textareaRef.current.style.height = "48px";
        textareaRef.current.focus();
      }
    });
  };

  const submit = (event) => {
    event.preventDefault();
    const text = value.trim();
    if (!text || disabled) return;

    onSend(text);
    setValue("");
    resetTextarea();
  };

  const resize = (element) => {
    element.style.height = "48px";
    element.style.height = `${Math.min(element.scrollHeight, 168)}px`;
  };

  return (
    <form className="message-input" onSubmit={submit}>
      <div className="message-input-field">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(event) => {
            setValue(event.target.value);
            resize(event.target);
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              submit(event);
            }
          }}
          placeholder="Sorunuzu yazın..."
          rows={1}
          disabled={disabled}
          aria-label="Sorunuzu yazın"
        />
      </div>
      <button type="submit" disabled={disabled || !value.trim()} title="Gönder">
        <span>Gönder</span>
      </button>
    </form>
  );
}
