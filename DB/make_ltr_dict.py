from collections import Counter

# .ltr 파일 경로
ltr_file_path = "/data1/hslim/PycharmProjects/hubert/DB/LibriSpeech/dev-clean.ltr"
# dict.ltr.txt 생성 경로
output_dict_path = "/data1/hslim/PycharmProjects/hubert/DB/LibriSpeech/dev-clean-dict.ltr.txt"

# .ltr 파일에서 문자 카운트
with open(ltr_file_path, "r") as f:
    lines = f.readlines()

# 모든 문자 추출 및 빈도 계산
counter = Counter()
for line in lines:
    counter.update(list(line.strip()))

# dict.ltr.txt 파일 생성
with open(output_dict_path, "w") as f:
    for char, freq in counter.items():
        f.write(f"{char} {freq}\n")

print(f"dict.ltr.txt 생성 완료: {output_dict_path}")
