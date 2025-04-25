import React, { useState } from "react";
import ChatInput from "../components/ChatInput";
import ChatWindow from "../components/ChatWindow";

const ChatRoute = () => {
  const [messages, setMessages] = useState([]);

  const handleSendMessage = async (message) => {
    // 사용자의 메시지 추가
    const newMessages = [...messages, { role: "user", content: message }];
    setMessages(newMessages);

    try {
      // 백엔드 요청
      const response = await fetch("http://localhost:32779/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: message }),
      });

      if (response.ok) {
        const data = await response.json();
        // AI 응답 추가
        setMessages((prevMessages) => [
          ...prevMessages,
          { role: "assistant", content: data.result },
        ]);
      } else {
        setMessages((prevMessages) => [
          ...prevMessages,
          { role: "assistant", content: "Error: Unable to process your query." },
        ]);
      }
    } catch (error) {
      console.error("Error:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { role: "assistant", content: "Error: Unable to connect to server." },
      ]);
    }
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        fontFamily: "Arial, sans-serif",
      }}
    >
      <ChatWindow messages={messages} />
      <ChatInput onSend={handleSendMessage} />
    </div>
  );
};

export default ChatRoute;