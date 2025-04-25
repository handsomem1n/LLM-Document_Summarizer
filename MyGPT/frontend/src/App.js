import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import ChatRoute from "./routes/ChatRoute";  // 채팅 화면
import AdminRoute from "./routes/AdminRoute"; // 어드민 화면

const App = () => {
  return (
    <Router>
      <Routes>
        {/* 채팅 화면 */}
        <Route path="/chat" element={<ChatRoute />} />
        
        {/* 어드민 화면 */}
        <Route path="/admin" element={<AdminRoute />} />
        
        {/* 기본 경로는 채팅 화면으로 리디렉션 */}
        <Route path="/" element={<ChatRoute />} />
      </Routes>
    </Router>
  );
};

export default App;
