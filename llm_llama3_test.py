import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# GPU 설정: 1번 GPU 사용
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 모델과 토크나이저 로드
model_name = r"C:\Users\nunuy\llm_data_m\llama_model_2"
print("Loading tokenizer and model...")

try:
    # 모델이 Hugging Face Hub에서 정상적으로 로드되는지 확인
    model_name = "beomi/Llama-3-Open-Ko-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")  # 모델을 GPU로 이동
    print("Hugging Face 모델 및 토크나이저 로딩 성공")
except Exception as e:
    print(f"모델 또는 토크나이저 로딩 중 오류 발생: {e}")
    exit()  # 로딩 오류 발생 시 프로그램 종료

# 문장 끝을 찾는 함수
def find_sentence_end(text):
    end_punctuations = [".", "!", "?"]
    for i in range(len(text) - 1, -1, -1):
        if text[i] in end_punctuations:
            return text[:i + 1]  # 문장 끝 구두점 다음까지 반환
    return text  # 구두점이 없으면 전체 텍스트 반환

# 요약 생성 함수
def generate_summary(text):
    # 요약을 유도하는 프롬프트
    prompt = f"Summarize the following text in Korean:\n{text}\n요약:"
    max_length = max(10, int(len(text) * 0.2))  # 최소 길이 10으로 설정
    min_length = max(5, int(max_length * 0.8))  # 최대 길이의 약 80%로 최소 길이 설정

    # 입력 텍스트를 토크나이즈하고 모델에 입력
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # GPU로 이동

    # 요약 생성
    try:
        print("Generating summary...")
        summary_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=100,  # 요약 생성 길이
            num_beams=4,
            no_repeat_ngram_size=2,
            length_penalty=1.0,
            repetition_penalty=1.5,
            early_stopping=True
        )
        generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).replace(prompt, "").strip()
        # 문장 끝을 구두점에서 끊기
        generated_summary = find_sentence_end(generated_summary)
    except Exception as e:
        generated_summary = f"요약 생성 중 오류 발생: {e}"

    return generated_summary

# 테스트 텍스트
test_text = "실증적 근거예산 전망의 객관성 확보를 위한 제도적 노력의 성과에 대해서는 다양한 견해가 존재한다. 견제를 담당하는 기관의 설립이 전망편의를 그다지 감소시키지 못한다는 연구결과가 있는가 하면, 독립 기관의 존재가 공식 전망의 GDP 예측 정확성을 향상시킨다는 의견도 있다. 하지만 분명한 것은, 많은 국가에서 비교 가능한 경험을 갖고 있으며 이러한 노력이 확산되고 있다는 점이다.미국 예산책임처(OMB)의 1947~2001년 세입 전망을 대상으로 한  연구는, 1974년 의회예산처(CBO) 설립 이후 OMB의 전망오차나 편의가 감소하였다고 보기 어렵다고 분석하였다. 그 원인은 양 기관이 경쟁보다는 협력의 전략을 선택함으로써 전망을 위한 공식적비공식적 정보를 공유하는 데 있다고 보았다.반면에 Frankel and Schreger(2012)105)는 17개 EU국가를 대상으로 한 연구에서, 독립재정기관이 예산에 대한 전망을 하는 유럽국가의 경우 그렇지 않은 국가에 비해 재정적자가 GDP대비 2% 이상 통계적으로 유의미하게 작게 나타나, 예산 전망이 더욱 정확하다는 분석 결과를 제시하였다. 이 연구를 확장한 IMF(2013)106)의 결과는 26개국, 1998~2010년간의 데이터를 대상으로 시행한 단순 합동회귀분석(pooled OLS)에서 재정위원회의 존재 및 동 기관의 중요한 특성들이 통계적으로 유의하게 예측오차를 줄일 수 있다는 분석결과를 내놓았다."

# 요약 생성 및 출력
print("Generated Summary:", generate_summary(test_text))
