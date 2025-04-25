import React, { useEffect, useState } from 'react';

const AdminPage = () => {
  const [conversation, setConversation] = useState([]);
  const [searchedDocuments, setSearchedDocuments] = useState([]);

  useEffect(() => {
    // 대화 기록을 가져오는 API 호출
    fetch('http://localhost:32779/admin/conversation')
      .then(response => response.json())
      .then(data => setConversation(data.conversation))
      .catch(error => console.error('Error fetching conversation:', error));

    // 검색된 문서들을 가져오는 API 호출
    fetch('http://localhost:32779/admin/searched_documents')
      .then(response => response.json())
      .then(data => setSearchedDocuments(data))
      .catch(error => console.error('Error fetching searched documents:', error));
  }, []);

  return (
    <div>
      <h1>Admin Page</h1>

      <h2>Conversation History</h2>
      <div>
        {conversation.map((msg, index) => (
          <div key={index}>
            <strong>{msg.role}</strong>: {msg.content}
          </div>
        ))}
      </div>

      <h2>Searched Documents</h2>
      <div>
        {searchedDocuments.map((doc, index) => (
          <div key={index}>
            <p>{doc}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AdminPage;
