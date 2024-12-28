from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os


def load_model(model_path):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Hugging Face token is not set in environment variables.")

    # 원본 모델 경로 (훈련 시 사용한 BASE_MODEL과 동일해야 함)
    base_model_path = "meta-llama/Llama-3.2-3B-Instruct"
    
    # 원본 모델 로드
    model = AutoModelForCausalLM.from_pretrained(base_model_path, use_auth_token=token)
    
    # LoRA 어댑터 로드
    model = PeftModel.from_pretrained(model, model_path)
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_auth_token=token)
    return model, tokenizer



def summarize_text(text, model, tokenizer):
    # 텍스트를 토큰화
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    
    # 모델을 이용해 요약 생성
    summary_ids = model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)
    
    # 요약된 텍스트 디코딩
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    # 저장된 모델과 토크나이저 로드
    model, tokenizer = load_model("results2/checkpoint-500")

    # 예시 텍스트
    text = '''
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

    # 요약 생성
    summary = summarize_text(text, model, tokenizer)
    print("Generated Summary:", summary)

if __name__ == "__main__":
    main()
