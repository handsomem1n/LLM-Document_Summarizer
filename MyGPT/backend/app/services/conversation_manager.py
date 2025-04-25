from typing import List, Dict
import ollama

class ConversationManager:
    def __init__(self, max_history_length=10):
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = max_history_length
        self.last_doc = []

    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})

        # 대화 기록이 최대 길이를 초과하면 요약
        if len(self.conversation_history) > self.max_history_length:
            self.summarize_history()

    def set_last_doc(self, docs:list):
        self.last_doc = docs

    def summarize_history(self):
        # 요약을 위한 대화 내용 텍스트 준비
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history])
        
        # Ollama API를 통해 요약 요청
        response = ollama.chat(model="bllossom", messages=[
            {"role": "user", "content": f"Summarize the following conversation:\n\n{history_text}"}
        ])
        
        # 요약본으로 대화 기록을 업데이트하고, 기존 기록은 삭제
        summary = response["message"]["content"]
        self.conversation_history = [{"role": "system", "content": summary}]

    def get_conversation(self):
        return self.conversation_history

    def reset_conversation(self):
        self.conversation_history = []

    def get_last_query(self):
        # 마지막 "user" 메시지를 반환
        for message in reversed(self.conversation_history):
            if message['role'] == 'user':
                return message['content']
        return None
    
    def get_last_doc(self):
        return self.last_doc
