import discord
import requests
import os

# Discord 봇 설정
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
API_URL = "http://localhost:32779/query"

class RAGDiscordBot(discord.Client):
    async def on_ready(self):
        print(f'Logged in as {self.user}')

    async def on_message(self, message):
        # 봇 자신이 보낸 메시지 무시
        if message.author == self.user:
            return

        # RAG 시스템에 사용자 쿼리 전송
        response = requests.post(API_URL, json={"query": message.content})
        
        if response.status_code == 200:
            result = response.json().get("result")
            await message.channel.send(result)
        else:
            await message.channel.send("An error occurred while processing your request.")

intents = discord.Intents.default()
intents.message_content = True

client = RAGDiscordBot(intents=intents)
client.run(DISCORD_TOKEN)
