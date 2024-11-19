import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from datasets import Dataset
import torch
from torch.utils.data import DataLoader

# llama 3.1 8B 모델과 토크나이저 로드
from transformers import LlamaForCausalLM, LlamaTokenizer

model_path = "/home/kookmin/kookmin/kd_8B/Llama3.1-8B"
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto", load_in_8bit=True)

# 토크나이저에 pad_token이 없으면 eos_token을 pad_token으로 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 데이터 로드 경로
passage_path = "/home/kookmin/kookmin/kd_8B/b_paper"
summary_path = "/home/kookmin/kookmin/kd_8B/s_paper"

# 데이터 로드 함수
def load_data(passage_dir, summary_dir):
    passages = []
    summaries = []
    for file_name in os.listdir(passage_dir):
        if file_name.endswith(".json"):
            # 입력 파일에서 passage_id 생성
            passage_id = file_name.replace(".json", "")
            
            # passage 파일 읽기
            with open(os.path.join(passage_dir, file_name), 'r', encoding='utf-8') as p_file:
                passage_data = json.load(p_file)
                passage_text = passage_data.get("Meta(Refine)", {}).get("passage", "")
            
            # 출력 파일 이름 생성 및 존재 여부 확인
            summary_file_path = os.path.join(summary_dir, file_name)
            if not os.path.exists(summary_file_path):
                print(f"Warning: {file_name}에 해당하는 summary 파일이 없어 건너뜁니다.")
                continue
            
            # summary 파일 읽기
            with open(summary_file_path, 'r', encoding='utf-8') as s_file:
                summary_data = json.load(s_file)
                summary_text = summary_data.get("summary1", "")
            
            # passage와 summary를 리스트에 추가
            passages.append(passage_text)
            summaries.append(summary_text)
                
    return passages, summaries

# 데이터 로드 및 준비
passages, summaries = load_data(passage_path, summary_path)
dataset = Dataset.from_dict({"text": passages, "summary": summaries})

# 데이터셋 전처리
def preprocess_data(examples):
    inputs = tokenizer(examples["text"], max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
    labels = tokenizer(examples["summary"], max_length=512, truncation=True, padding="max_length", return_tensors="pt").input_ids
    inputs['labels'] = labels
    return inputs

# 데이터셋 전처리 실행
dataset = dataset.map(preprocess_data, batched=True)

# collate_fn 함수 정의
def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    labels = torch.stack([torch.tensor(item['labels']) for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# DataLoader로 데이터셋을 배치 형태로 로드
batch_size = 8  # 원하는 배치 크기로 설정
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# 모델 학습 및 저장 함수
def train_and_save(model, dataloader, save_path="/home/kookmin/kookmin/kd_8B/llama_3.1_8B_model", num_epochs=3, learning_rate=5e-5):
    model.train()
    
    # 옵티마이저 설정
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0
        for batch in dataloader:
            # 배치 데이터를 모델 입력으로 설정
            inputs = {
                'input_ids': batch['input_ids'].to(model.device),
                'attention_mask': batch['attention_mask'].to(model.device),
                'labels': batch['labels'].to(model.device)
            }

            # 모델의 그라디언트 초기화 및 순전파 + 역전파
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            # 손실을 기록
            total_loss += loss.item()
            print(f"Loss: {loss.item()}")

        avg_loss = total_loss / len(dataloader)
        print(f"Average loss for epoch {epoch + 1}: {avg_loss}")
    
    # 학습이 끝난 모델 저장
    model.save_pretrained(save_path)    
    tokenizer.save_pretrained(save_path)
    print("모델과 토크나이저가 성공적으로 저장되었습니다.")

# 모델 학습 및 저장 실행
train_and_save(model, dataloader, save_path="/home/kookmin/kookmin/kd_8B/llama_3.1_8B_model")
