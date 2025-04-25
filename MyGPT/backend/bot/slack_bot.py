import os
import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# 환경 변수에서 토큰과 API URL 불러오기
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")  # Socket Mode 앱 토큰
API_URL = "http://localhost:32779/query"

# Slack Bolt 앱 초기화
app = App(token=SLACK_BOT_TOKEN)

# 메시지 이벤트 핸들러 설정
@app.message("")  # 모든 메시지를 수신
def handle_message_events(message, say):
    user_message = message['text']
    channel_id = message['channel']
    print(channel_id, user_message)
    
    # RAG 시스템에 사용자 쿼리 전송
    response = requests.post(API_URL, json={"query": user_message})
    
    if response.status_code == 200:
        result = response.json().get("result")
        say(text=result, channel=channel_id)
    else:
        say(text="An error occurred while processing your request.", channel=channel_id)

# Socket Mode 핸들러 초기화 및 실행
if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
