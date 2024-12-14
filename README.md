# LLM-Document_Summarizer
LLM기반 전자문서 요약 자동 생성 모델 - 국민대학교x가톨릭관동대학교 대학원

# 프로젝트 기간
2024/10/26 ~ 2024/11/30

# 구성원
국민대학교 학부생 5명 - Llama3.2 3B instruct Model 담당
가톨릭관동대 석사생 2명 - Llama3.1 8B instruct Model 담당

# Branch 설명
- deployment가 최종적이며, 해당 Branch에는 자체 내부적으로 선정한 모델 - Llama3.2 3B Instruct입니다. 

# **자동화된 PDF 요약 및 검색 시스템**

이 프로젝트는 PDF 파일의 텍스트를 추출하고 LLM(Large Language Model)을 활용하여 요약문을 생성하며 검색 가능한 요약 데이터를 저장하는 파이프라인을 제공합니다. RAG(Retrieval-Augmented Generation) 기법을 통해 효율적인 검색 및 응답 기능을 구현하였습니다.

자세한 작동 방식은 아래의 diagram을 참고하세요.

# System Design Diagrams

| Workflow | Data Flow Diagram (DFD) | Sequence Diagram |
|----------|--------------------------|------------------|
| ![Workflow](https://github.com/user-attachments/assets/9d78216d-08e1-44c4-9208-b71281247509) | ![DFD](https://github.com/user-attachments/assets/200489ab-0caa-4c88-93be-64130e28b477) | ![Sequence Diagram](https://github.com/user-attachments/assets/eef9831b-9a79-403a-838e-d6f3e61c2a65) |




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
언어: Python
모델: Llama3.2 (LoRA 적용)
데이터 처리: pdfplumber, json
검색 기술: RAG (Chroma 벡터 스토어)
