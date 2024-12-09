import os

# 경로 설정
root_dir = "/data1/hslim/PycharmProjects/hubert/DB/LibriSpeech/dev-clean"  # train-clean-100 디렉토리
output_ltr_file = "/data1/hslim/PycharmProjects/hubert/DB/LibriSpeech/dev-clean.ltr"  # 최종 .ltr 파일 경로

# 결과를 저장할 리스트
lines = []

# 모든 하위 디렉토리 순회
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".txt"):  # .txt 파일만 처리
            txt_path = os.path.join(subdir, file)
            with open(txt_path, "r") as f:
                for line in f:
                    line = line.strip().lower()  # 소문자로 변환
                    line = line.replace(" ", "|")  # 공백을 "|"로 대체
                    lines.append(line)

# 결과를 .ltr 파일로 저장
with open(output_ltr_file, "w") as f:
    f.write("\n".join(lines))

print(f"LTR 파일 생성 완료: {output_ltr_file}")
