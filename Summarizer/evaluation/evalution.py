import json
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
import nltk
import pandas as pd

# NLTK 리소스 다운로드
nltk.download('wordnet')
nltk.download('omw-1.4')

# JSON 파일 읽기
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# BLEU 점수 계산
def calculate_bleu(reference, candidate):
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    return sentence_bleu(reference_tokens, candidate_tokens)

# ROUGE 점수 계산
def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }

# BERTScore 계산
def calculate_bertscore(reference, candidate):
    # BERTScore 계산
    P, R, F1 = bert_score([candidate], [reference], lang="ko", rescale_with_baseline=True)
    return F1.mean().item()  # F1 평균 반환

# METEOR 점수 계산
def calculate_meteor(reference, candidate):
    # reference와 candidate를 공백 기준으로 토큰화
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    # METEOR 점수 계산
    return meteor_score([reference_tokens], candidate_tokens)

# 평가 파이프라인
def evaluate_summaries(json_data):
    results = []
    for item in json_data:
        passage = item["passage"]
        summary = item["summary"]
        
        # BLEU, ROUGE, BERTScore, METEOR 계산
        bleu_score = calculate_bleu(passage, summary)
        rouge_scores = calculate_rouge(passage, summary)
        bert_score_val = calculate_bertscore(passage, summary)
        meteor_score_val = calculate_meteor(passage, summary)

        results.append({
            "id": item["id"],
            "bleu": bleu_score,
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"],
            "bertscore": bert_score_val,
            "meteor": meteor_score_val
        })

    return pd.DataFrame(results)

# 실행 예시
if __name__ == "__main__":
    input_file = "evaluation/summarized_data.json"  # 입력 JSON 파일 경로
    data = load_json(input_file)

    # 평가 수행
    evaluation_results = evaluate_summaries(data)
    print(evaluation_results)

    # # 평가 결과 저장
    # evaluation_results.to_csv("evaluation/evaluation_results.csv", index=False, encoding="utf-8")
    # print("평가 결과가 저장되었습니다.")
