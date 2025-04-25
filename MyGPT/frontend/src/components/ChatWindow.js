import React from "react";
import LoadingAnimation from "./loadingAnimation";

const ChatWindow = ({ messages, loading }) => {
  return (
    <div className="chat-window">
      {messages.map((msg, index) => (
        <div
          key={index}
          style={{ textAlign: msg.role === "user" ? "right" : "left", marginBottom: "10px" }}
        >
          <div className={`chat-bubble ${msg.role}`}>{msg.content}</div>
        </div>
      ))}
      {loading && <LoadingAnimation />}
    </div>
  );
};

export default ChatWindow;
