# LLM-Document_Summarizer
LLM기반 경량화 기법을 적용한 전자결재 문서 생성 요약

# 논문
해당 프로젝트는 연구로서의 가치를 인정받아 논문을 투고로 이어졌고, DBpia에서 열람 가능합니다.

Link : https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12132735

# Introduction
<img width="1148" alt="image" src="https://github.com/user-attachments/assets/7c6ceeb9-a04f-4091-853e-ac9794bd2992" />


# Research Background
<img width="1147" alt="image" src="https://github.com/user-attachments/assets/c2b8546c-17d3-4fe1-835c-0d5f574553c3" />

# Experimental Dataset
<img width="1146" alt="image" src="https://github.com/user-attachments/assets/c29570db-a0b8-46a3-80dd-ef44ba2c4bb2" />

# Lightweight Model Optimization
<img width="1145" alt="image" src="https://github.com/user-attachments/assets/d54f2908-4738-4d92-9694-f89341058d81" />

# Evaluation Metrics
<img width="1145" alt="image" src="https://github.com/user-attachments/assets/5b729e51-45e2-4889-acda-304f05126027" />

# Score Result
<img width="1147" alt="image" src="https://github.com/user-attachments/assets/4db14fa6-d85f-4009-a7b9-825eacb8114b" />

# Summarization Result - Llama3.2 3B(Quantzation + LoRA)
<img width="1145" alt="image" src="https://github.com/user-attachments/assets/da2510b3-1225-4d68-8409-8c81c6fccb67" />

# Improve Measures
<img width="1145" alt="image" src="https://github.com/user-attachments/assets/375c5fff-829e-49e8-9878-c04c65227a59" />










# 프로젝트 기간
2024/10/26 ~ 2024/11/30

# 구성원
국민대학교 학부생 5명 - Llama3.2 3B instruct Model 담당
가톨릭관동대 석사생 2명 - Llama3.1 8B instruct Model 담당

# Branch 설명
- deployment가 최종 레포지토리이며, 해당 Branch에는 가장 우수한 성능을 보인 모델 - Llama3.2 3B Instruct입니다.

---
# 시연 자료 - 테스트 가이드
[llama3.2-3B-테스트가이드.pdf](https://github.com/user-attachments/files/20116361/llama3.2-3B-.pdf)

---

## **📌 주요 기능**
1. **PDF 파일 처리**  
   - PDF 파일에서 텍스트와 메타데이터를 추출.
   - 추출된 데이터를 JSON 형식으로 저장.

2. **요약 생성**  
   - LLM(Llama3.2 모델)을 활용하여 긴 텍스트를 요약.
   - 도메인에 적합한 요약문 생성.

3. **검색 기능**  
   - RAG 기법을 사용하여 요약 데이터를 벡터화.
   - 사용자 쿼리에 대해 연관 문서를 검색하고 응답 생성.

4. **평가 지표**  
   - **BLEU**: n-gram 기반 평가로 요약의 정밀도를 측정.
   - **BERTScore**: 문맥적 유사성을 기반으로 요약 품질 평가.

---

## **🚀 설치 및 실행 방법**

### **1. 의존성 설치**
pip install -r requirements.txt

2. PDF 파일 처리
PDF 파일을 특정 디렉토리에 배치한 후 아래 명령을 실행하십시오:
python pdf_processor.py --input-dir ./pdfs --output-dir ./jsons


3. 요약 생성
추출된 JSON 파일을 바탕으로 Llama3.2 모델을 사용하여 요약을 생성하십시오:
python summarize.py --input-dir ./jsons --output-dir ./summaries

4. 검색 및 응답 시스템 실행
RAG 기반 검색 시스템을 실행하려면 다음 명령을 실행하십시오:
python rag_service.py

## 📊 평가 결과
| 모델 ID          | BLEU   | BERTScore | 학습 도메인 |
|------------------|--------|-----------|-------------|
| 제공된 요약 샘플 | 0.4644 | 0.7197    | -           |
| Llama3.1 - 8B   | 0.1306 | 0.6906    | 뉴스        |
| Llama3.2 - 3B   | 0.2618 | 0.7466    | 보고서      |


<img width="578" alt="image" src="https://github.com/user-attachments/assets/c0bbc6c8-0507-48fe-9cbb-75dd4b67a287">


## 🛠️ 기술 스택
| **항목**          | **설명**               |
|-------------------|-----------------------|
| **언어**          | Python               |
| **모델**          | Llama3.2 (LoRA 적용) |
| **데이터 처리**    | pdfplumber, json     |
| **검색 기술**      | RAG (Chroma 벡터 스토어) |



자세한 프로젝트의 아래의 diagram을 참고하세요.

# System Design Diagrams

| Workflow | Data Flow Diagram (DFD) | Sequence Diagram |
|----------|--------------------------|------------------|
| ![Workflow](https://github.com/user-attachments/assets/9d78216d-08e1-44c4-9208-b71281247509) | ![DFD](https://github.com/user-attachments/assets/200489ab-0caa-4c88-93be-64130e28b477) | ![Sequence Diagram](https://github.com/user-attachments/assets/eef9831b-9a79-403a-838e-d6f3e61c2a65) |




