import os


def create_tsv_exclude_test_clean(root_dir, output_tsv_path):
    # TSV 파일 생성
    with open(output_tsv_path, "w") as tsv_file:
        # 루트 디렉토리 작성
        tsv_file.write(f"{root_dir}\n")

        # 디렉토리 내 모든 파일 탐색
        for root, _, files in os.walk(root_dir):
            # test-clean 디렉토리를 제외
            if "test-clean" in root:
                continue

            for file in files:
                # .wav 파일만 처리
                if file.endswith(".wav"):
                    # 파일 경로 작성
                    audio_path = os.path.relpath(os.path.join(root, file), start=root_dir)
                    tsv_file.write(f"{audio_path}\n")


# 디렉토리 경로와 TSV 파일 경로 설정
root_dir = "/data1/hslim/PycharmProjects/hubert/DB/LibriSpeech/dev-clean"
output_tsv_path = "/data1/hslim/PycharmProjects/hubert/DB/LibriSpeech/dev_tsv_file.tsv"

# TSV 생성 함수 호출
create_tsv_exclude_test_clean(root_dir, output_tsv_path)

print(f"TSV 파일이 {output_tsv_path}에 생성되었습니다!")
