import os
import json

def extract_and_save_json(input_folder, output_file):
    # 폴더 내 모든 파일을 순회
    extracted_data = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 필요한 필드 추출
                extracted_entry = {
                    "doc_id": data["Meta(Acqusition)"]["doc_id"],
                    "doc_type": data["Meta(Acqusition)"]["doc_type"],
                    "doc_name": data["Meta(Acqusition)"]["doc_name"],
                    "passage": data["Meta(Refine)"]["passage"],
                    "summary": data["Annotation"]["summary1"]
                }
                extracted_data.append(extracted_entry)

    # 추출된 데이터를 단일 JSON 파일로 저장
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=4)

# 사용 예시
input_folder = "/workspace/TRdata_paper/01.데이터/1.Training/라벨링데이터/TL1/04.paper/20per"
output_file = "/workspace/TRdata_paper/01.데이터/1.Training/라벨링데이터/TL1/04.paperextracted_data.json"
extract_and_save_json(input_folder, output_file)