import os
import re
import torch
import itertools
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from bert_score import score

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

# 모델과 토크나이저 로드 함수
def load_model(model_path):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Hugging Face token is not set in environment variables.")

    base_model_path = "meta-llama/Llama-3.2-3B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, token=token, device='cuda')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    
    model = AutoModelForCausalLM.from_pretrained(base_model_path, token=token).to('cuda')
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, model_path, device='cuda')
    
    return model, tokenizer

# 텍스트 요약 함수
def summarize_text(text, model, tokenizer, max_new_tokens, temperature, no_repeat_ngram_size, num_beams):
    inputs = tokenizer(text, return_tensors="pt").to('cuda')
    summary_ids = model.generate(
        inputs['input_ids'],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams,
        early_stopping=True
    )
    decoded_output = tokenizer.batch_decode(summary_ids, skip_special_tokens=False)[0]
    filtered_output = decoded_output.replace(text, '').strip()
    return filtered_output

# BERTScore 계산 함수
def calculate_bertscore(summary, reference):
    P, R, F1 = score([reference], [summary], lang="ko", rescale_with_baseline=True)
    return F1.mean().item()

# Grid Search 함수
def grid_search(text, reference, model, tokenizer, max_len):
    max_new_tokens_options = [max_len]
    temperature_options = [0.3, 0.4, 0.5]
    num_beams_options = [4, 5, 6]
    no_repeat_ngram_size_options = [3, 5, 7, 9, 11]
    # repetition_penalty_options = [1.0, 1.2, 1.5]

    best_score = 0
    best_params = None

    for max_new_tokens, temperature, num_beams, no_repeat_ngram_size in itertools.product(
        max_new_tokens_options, temperature_options, num_beams_options, no_repeat_ngram_size_options
    ):
    # for max_new_tokens, temperature, num_beams, no_repeat_ngram_size in itertools.product(
    #     max_new_tokens_options, temperature_options, no_repeat_ngram_size_options
    # ):
    # for max_new_tokens, temperature, no_repeat_ngram_size in itertools.product(
    #     max_new_tokens_options, temperature_options, no_repeat_ngram_size_options
    # ):
        summary = summarize_text(
            text, model, tokenizer, max_new_tokens, temperature, no_repeat_ngram_size, num_beams
        )
        score = calculate_bertscore(summary, reference)
        
        if score > best_score:
            best_score = score
            best_params = (max_new_tokens, temperature, no_repeat_ngram_size, num_beams)
        
        print(f"Params: {max_new_tokens}, {temperature}, {num_beams}, {no_repeat_ngram_size} => BERTScore: {score}")

    return best_params, best_score

# 한자 및 일본어 제거 함수
def remove_non_korean(text):
    return re.sub(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]+', '', text)

# 메인 함수
def main():
    model, tokenizer = load_model("results3/checkpoint-1000")

    inputs_raw = "철저한 원가 분석을 통해 원가에 미달하는 가격규제로 인해 부채가 발생한 부분과 그렇지 않는 부분을 구분하여 규명해야 함. 셋째, 정부 주도 정책들로서 먼저 공공기관부터 시범 실시해보라고 떠넘긴 경우.－ 주로 임금과 노동 관련 정책들의 경우임. 예를 들어 근로자 복지 기본법이 2010년 제정 되어 선택적 근로자 복지 제도가 생겨남.－ 이른바 카페테리아 플랜이라는 제도가 복리후생의 핵심이었는데 민간기업이 선뜻 채택을 하지 않자 공기업부터 먼저 실시해보라고 함.－ 또 정년연장, 임금피크제 등 정부 정책을 공기업부터 실시해보라고 해서 생긴 원가의 상승 압력이 분명히 있음.  넷째, 자체 부실 문제. 부채의 원인을 위 네 가지 유형으로 구분해서 부채의 원인 규명해야 함.  그러므로 구분회계부터 먼저 시작을 해야 함.－ 현재 부채 493조 중 일부는 발생 원인을 명확히 규명할 수 없을지도 모름. － 그러나 각 기관별로 상당부문은 각 사업의 어떤 부분이 어떤 원인에 의해 발생했는지 귀책사유를 밝히는 것이 가능하다고 봄. 국가 부채의 전반적인 관리시스템 혁신 우리나라는 OECD 가입 국가임에도 국가부채 통계를 1년에 딱 한 번 그것도 5월에 발표함.－ 우리는 1년에 한 번 그나마 전년도의 부채를 다음해 5월에 발표하므로 지금 우리가 사용하는 국가부채 통계는 2012년 기준임.－ 게다가 1년 동안 어떻게 변화되어 왔는지 알 수 있는 통계자료가 발표되지 않고 있음.  미국의 경우 뉴욕 타임스퀘어에 국가부채시계(National Debt Clock)를 설치하여 현재 국가 부채가 얼마인지 알려주고, 웹 사이트를 통해 실시간으로 변동되는 공공부채규모를 각 부문별로 보여주고 있음."

    reference_summary = "국회예산정책처의 발제는 공공기관 재무건강 악화에 대한 정확한 진단과 관리체계의 문제점이 지적되었다. 2012년 부채상위 10개 공기업 부채 규모가 425조원으로 전체 공공기업의 86% 수준이며 304개 전체 공공기업 전반에 걸친 문제라고 파악하는 것은 논란이 있다."

    summarize_len = len(inputs_raw) // 100 * 10
    prompt = f"""
    MAKE SURE THAT YOU SUMMARIZE THE FOLLOWING TEXT TO A MAXIMUM OF {summarize_len} TOKENS. THE SUMMARY CAN BE SHORTER if all essential information is included, ensuring the following rules:
n
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
    inputs = f"<|begin_of_text|>{prompt}\n{inputs_raw}<|eot_id|>"
    best_params, best_score = grid_search(inputs, inputs_raw, model, tokenizer, summarize_len)
    print(f"Best Params: {best_params} => Best BERTScore: {best_score}")


if __name__ == "__main__":
    main()
