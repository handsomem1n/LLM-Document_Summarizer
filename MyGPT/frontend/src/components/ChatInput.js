import React, { useState } from "react";

const ChatInput = ({ onSend }) => {
  const [message, setMessage] = useState("");

  const handleSend = () => {
    if (message.trim()) {
      onSend(message);
      setMessage("");
    }
  };

  return (
    <div style={{ display: "flex", padding: "10px", borderTop: "1px solid #ccc" }}>
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Type your message..."
        style={{ flex: 1, padding: "10px", fontSize: "16px" }}
      />
      <button
        onClick={handleSend}
        style={{
          marginLeft: "10px",
          padding: "10px 20px",
          fontSize: "16px",
          background: "#007BFF",
          color: "#fff",
          border: "none",
          borderRadius: "5px",
          cursor: "pointer",
        }}
      >
        Send
      </button>
    </div>
  );
};

export default ChatInput;
