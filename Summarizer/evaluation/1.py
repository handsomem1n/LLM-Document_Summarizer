from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os
import re
from transformers import StoppingCriteria, StoppingCriteriaList
from bert_score import score
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 사용자 정의 StoppingCriteria
class StopOnKeyword(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, max_words=500):
        self.stop_words = stop_words
        self.tokenizer = tokenizer
        self.words = 0
        self.max = max_words

    def __call__(self, input_ids, scores):
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        self.words += 1
        if self.words > self.max:
            return any(word in generated_text[-1] for word in self.stop_words)
        return False



def load_model(model_path):
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("Hugging Face token is not set in environment variables.")

    # 원본 모델 경로 (훈련 시 사용한 BASE_MODEL과 동일해야 함)
    base_model_path = "meta-llama/Llama-3.2-3B-Instruct"

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, token=token, device='cuda')

    # pad_token 설정
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # 원본 모델 로드
    model = AutoModelForCausalLM.from_pretrained(base_model_path, token=token).to('cuda')

    # 모델과 토크나이저 동기화
    model.resize_token_embeddings(len(tokenizer))

    # LoRA 어댑터 로드
    model = PeftModel.from_pretrained(model, model_path, device='cuda')
    return model, tokenizer

def summarize_text(text, model, tokenizer, max_new_tokens, temperature, num_beams, no_repeat_ngram_size, repetition_penalty, stopping_criteria):
    # 텍스트를 토큰화
    # 입력 토큰화
    inputs = tokenizer(text, return_tensors="pt").to('cuda')

    # 출력 생성
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        eos_token_id=128009,
        temperature=temperature,
        no_repeat_ngram_size=no_repeat_ngram_size,  # 반복을 방지
        repetition_penalty = repetition_penalty,
        stopping_criteria=stopping_criteria,  # 사용자 정의 종료 조건
        # early_stopping=True,
        num_beams=num_beams,
        do_sample=True
    )

    # 불필요한 특수 문자 제거 및 포맷팅
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
    filtered_output = decoded_output.replace(text, '').strip()

    # 한자 및 일본어 제거 함수
    def remove_non_korean(text):
        """
        한자(중국어) 및 일본어를 제거하는 함수.
        유니코드 범위:
        - 한자: \u4E00-\u9FFF
        - 일본어(히라가나): \u3040-\u309F
        - 일본어(가타카나): \u30A0-\u30FF
        """
        return re.sub(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]+', '', text)



    # 불필요한 문자 제거 및 최종 출력
    final_output = remove_non_korean(filtered_output)
    return final_output

# bertscore 계산
def calculate_bertscore(summary, reference):
    P, R, F1 = score([summary], [reference], lang="ko", verbose=True) 
    return F1.mean().item()

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ grid search @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def grid_search(text, model, tokenizer, summary_len, stopping_criteria):
    total_score = []
    
    max_new_tokens_options = [summary_len]
    
    # 1
    temperature_options = [0.3, 0.4, 0.5]
    
    # 2
    no_repeat_ngram_size_options = [3, 4, 5, 6,7]
    
    # 3
    num_beams_options = [4, 5, 6]
    
    # 4
    repetition_penalty_options = [1.0, 1.1, 1.2, 1.3, 1.5]

    best_score = 0
    best_params = None

    for max_new_tokens, temperature, num_beams, no_repeat_ngram_size, repetition_penalty in itertools.product(
        max_new_tokens_options, temperature_options, num_beams_options, no_repeat_ngram_size_options, repetition_penalty_options
    ):
        summary = summarize_text(
            text, model, tokenizer, max_new_tokens, temperature, num_beams, no_repeat_ngram_size, repetition_penalty, stopping_criteria
        )
        score = calculate_bertscore(summary, text)

        if score > best_score:
            best_score = score
            best_params = (max_new_tokens, temperature, num_beams, no_repeat_ngram_size, repetition_penalty)
        total_score.append({score: (max_new_tokens, temperature, num_beams, no_repeat_ngram_size, repetition_penalty)})

        print(f"Params: {max_new_tokens}, {temperature}, {num_beams}, {no_repeat_ngram_size}, {repetition_penalty} => BERTScore: {score}")

    # 점수 데이터를 행렬로 변환
    score_matrix = np.zeros((
        len(temperature_options),
        len(no_repeat_ngram_size_options),
        len(num_beams_options),
        len(repetition_penalty_options)
    ))
    
    # total_score의 데이터를 행렬에 채우기
    for score_dict in total_score:
        for score, params in score_dict.items():
            _, temp, beam, ngram, rep = params
            i = temperature_options.index(temp)
            j = no_repeat_ngram_size_options.index(ngram)
            k = num_beams_options.index(beam)
            l = repetition_penalty_options.index(rep)
            score_matrix[i, j, k, l] = score
    
    # 4D 행렬을 2D로 펼치기
    n_rows = len(temperature_options) * len(no_repeat_ngram_size_options)
    n_cols = len(num_beams_options) * len(repetition_penalty_options)
    heatmap_data = np.zeros((n_rows, n_cols))
    
    for i in range(len(temperature_options)):
        for j in range(len(no_repeat_ngram_size_options)):
            for k in range(len(num_beams_options)):
                for l in range(len(repetition_penalty_options)):
                    row = i * len(no_repeat_ngram_size_options) + j
                    col = k * len(repetition_penalty_options) + l
                    heatmap_data[row, col] = score_matrix[i, j, k, l]
    
    # 히트맵 그리기
    plt.figure(figsize=(15, 10))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd')
    
    # 축 레이블 설정
    row_labels = [f'T:{t}, N:{n}' for t in temperature_options for n in no_repeat_ngram_size_options]
    col_labels = [f'B:{b}, R:{r}' for b in num_beams_options for r in repetition_penalty_options]
    
    plt.xticks(np.arange(len(col_labels)) + 0.5, col_labels, rotation=45, ha='right')
    plt.yticks(np.arange(len(row_labels)) + 0.5, row_labels, rotation=0)
    
    plt.title('Grid Search Score Heatmap')
    plt.tight_layout()
    plt.savefig('grid_search_heatmap.png')
    plt.close()

    return best_params, best_score, total_score



def main():
    # 저장된 모델과 토크나이저 로드
    model, tokenizer = load_model("/workspace/workspace/1129backup/LLM-Document_Summarizer/results/checkpoint-27534")

    # 예시 텍스트
    inputs_raw = '''
    철저한 원가 분석을 통해 원가에 미달하는 가격규제로 인해 부채가 발생한 부분과 그렇지 않는 부분을 구분하여 규명해야 함.
    셋째, 정부 주도 정책들로서 먼저 공공기관부터 시범 실시해보라고 떠넘긴 경우.－ 주로 임금과 노동 관련 정책들의 경우임.
    예를 들어 근로자 복지 기본법이 2010년 제정 되어 선택적 근로자 복지 제도가 생겨남.－
    이른바 카페테리아 플랜이라는 제도가 복리후생의 핵심이었는데 민간기업이 선뜻 채택을 하지 않자 공기업부터 먼저 실시해보라고 함.－
    또 정년연장, 임금피크제 등 정부 정책을 공기업부터 실시해보라고 해서 생긴 원가의 상승 압력이 분명히 있음.
    넷째, 자체 부실 문제. 부채의 원인을 위 네 가지 유형으로 구분해서 부채의 원인 규명해야 함.
    그러므로 구분회계부터 먼저 시작을 해야 함.－ 현재 부채 493조 중 일부는 발생 원인을 명확히 규명할 수 없을지도 모름.
    － 그러나 각 기관별로 상당부문은 각 사업의 어떤 부분이 어떤 원인에 의해 발생했는지 귀책사유를 밝히는 것이 가능하다고 봄.
    국가 부채의 전반적인 관리시스템 혁신 우리나라는 OECD 가입 국가임에도 국가부채 통계를 1년에 딱 한 번 그것도 5월에 발표함.
    우리는 1년에 한 번 그나마 전년도의 부채를 다음해 5월에 발표하므로 지금 우리가 사용하는 국가부채 통계는 2012년 기준임.
    게다가 1년 동안 어떻게 변화되어 왔는지 알 수 있는 통계자료가 발표되지 않고 있음.
    미국의 경우 뉴욕 타임스퀘어에 국가부채시계(National Debt Clock)를 설치하여 현재 국가 부채가 얼마인지 알려주고, 웹 사이트를 통해 실시간으로 변동되는 공공부채규모를 각 부문별로 보여주고 있음.
    '''

    # prompt 생성
    summary_len = len(inputs_raw) // 100 * 10
    prompt = f"""
    MAKE SURE THAT YOU SUMMARIZE THE FOLLOWING TEXT TO A MAXIMUM OF {summary_len} TOKENS. THE SUMMARY CAN BE SHORTER if all essential information is included, ensuring the following rules:

    1. **Summary Quality:**
    - The summarized text should have no spelling errors or typos.
    - Avoid repeating similar content. If multiple sentences convey similar ideas, output only one concise sentence to represent them.
    - The text should be logically structured and divided into appropriate paragraphs to maintain readability.

    2. **Key Information:**
    - Ensure that the summary includes key points such as the causes of debt, government policies, and the need for improved debt management systems.

    3. **Prevent Duplication:**
    - Do not generate sentences that repeat or convey the same idea as other sentences within the summarized text.
    - If a sentence shares a similar meaning with another, only include the most concise and representative one. The rest must be omitted.
    - The summary does not need to reach a specific target length, as long as all essential information is included without duplication.

    4. **Example of a Good Summary:**
    - "The analysis highlights the need to distinguish between debt caused by price regulations and other factors, emphasizing government policy impacts and the necessity for better debt management."

    5. **Avoid This Type of Summary:**
    - "Debt is caused by many things. Government policies are involved. Management is needed." (Too vague and lacks detail)

    Ensure the final summarized text adheres to these rules and retains its readability and logical structure.
    """

    inputs_raw = f"""<|begin_of_text|><|start_header_id|>user: <|end_header_id|>{prompt}
    {inputs_raw}<|eot_id|><|start_header_id|>assistant: <|end_header_id|>
    """

    # 종료 조건 설정
    stop_words = ["."]  # 종료를 트리거하는 키워드
    stopping_criteria = StoppingCriteriaList([StopOnKeyword(stop_words, tokenizer, summary_len)])

    best_params, best_score, total_score = grid_search(inputs_raw, model, tokenizer, summary_len, stopping_criteria)
    print(f"Best Params: {best_params}, Best Score: {best_score}")
    

    

if __name__ == "__main__":
    main()