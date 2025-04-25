import React from "react";
import "../style.css";

const LoadingAnimation = () => {
  return (
    <div className="typing-animation">
      <span>메시지 작성 중</span>
      <div className="typing-dots">
        <span></span>
        <span></span>
        <span></span>
      </div>
    </div>
  );
};

export default LoadingAnimation;
