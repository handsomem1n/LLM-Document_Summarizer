import pdfplumber
import os
import json
import re

def extract_sections_with_pdfplumber(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text()

    main_title = ""
    title = ""
    content_section = ""
    attachment = ""
    finishline_section = ""
    pass_date = ""

    # 큰제목 추출: "수신"이 포함된 줄의 이전 줄을 큰제목으로 설정
    if "수신" in full_text:
        lines = full_text.splitlines()  # 전체 텍스트를 줄 단위로 분리
        for i, line in enumerate(lines):
            if "수신" in line:
                if i > 0:  # 수신이 포함된 줄의 이전 줄이 존재하는 경우
                    main_title = lines[i - 1].strip()
                break

    # 제목 추출: "제목" 이후의 내용을 제목으로 설정
    title_start = full_text.find("제목") + len("제목")
    title_end = full_text.find("\n", title_start)
    if title_start != -1 and title_end != -1:
        title = full_text[title_start:title_end].strip()

    # 본문 추출: "1." 이후 본문의 시작과 "붙임 " 이전의 본문 내용만 가져옴
    content_start = full_text.find(title) + len(title)
    content_end = full_text.find("붙임 ")
    if content_end == -1:
        content_end = full_text.find("끝.")

    if content_start != -1 and content_end != -1:
        content_section = full_text[content_start:content_end].strip()

    # 붙임 파일 이름 추출
    attachment_start = full_text.find("붙임 ")
    attachment_end = full_text.find("끝.")
    if attachment_start != -1:
        attachment_line = full_text[attachment_start:attachment_end].strip()
        attachment = attachment_line.replace("붙임 ", "").strip()  # "붙임 " 제거
        attachment = attachment.replace(":", "").strip()
    else: 
        attachment = "없음"

    # 결문부분 본문의 끝. 뒤부터 협조자 앞까지만 추출
    finishline_start = full_text.find("끝.") + len("끝.")
    finishline_end = full_text.find("협조자")
    if finishline_start != -1 and finishline_end != -1:
        finishline_section = full_text[finishline_start:finishline_end].strip()

        # 날짜 추출: "2"로 시작하고 첫 번째 점으로 끝나는 부분
        pass_date = ""
        date_pattern_start = finishline_section.find("2")
        if date_pattern_start != -1:
            first_dot = finishline_section.find('.', date_pattern_start)
            if first_dot != -1:
                year = finishline_section[date_pattern_start:first_dot].strip()
                second_dot = finishline_section.find('.', first_dot + 1)
                if second_dot != -1:
                    month = finishline_section[second_dot - 2:second_dot].strip()
                    third_dot = finishline_section.find('.', second_dot + 1)
                    if third_dot != -1:
                        day = finishline_section[third_dot - 2:third_dot].strip()
                        pass_date = f"{year}.{month}.{day}"
                        pass_date = pass_date.replace(" ", "").strip()

        # 추출한 날짜 제거 및 \n 삭제
        if pass_date:
            finishline_section = finishline_section.replace(year, "").replace(month, "").replace(day, "").replace("\n", "").replace(".", "").strip()
            

    # JSON 데이터 생성
    data = {
        "Meta(Refine)": {
            "passage_id": os.path.basename(pdf_path),
            "교육청": main_title,
            "제목": title,
            "본문": content_section,
            "붙임파일": attachment,
            "승인자": finishline_section,
            "승인날짜": pass_date
        }
    }
    return data

def save_to_json(data, json_path):
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

def process_pdfs_in_folder(folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            extracted_data = extract_sections_with_pdfplumber(pdf_path)
            json_filename = os.path.splitext(filename)[0] + ".json"
            json_path = os.path.join(output_folder, json_filename)
            save_to_json(extracted_data, json_path)
            print(f"Extracted and saved: {json_filename}")

if __name__ == "__main__":
    folder_path = r"/Users/newuser/temp/pdf_to_json"
    output_folder = r"/Users/newuser/temp/pdf_to_json"
    process_pdfs_in_folder(folder_path, output_folder)
    print("All PDFs processed and saved to JSON.")