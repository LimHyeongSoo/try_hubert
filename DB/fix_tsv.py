import os
import soundfile as sf

# 기존 .tsv 파일 경로
input_tsv_path = "/data1/hslim/PycharmProjects/hubert/DB/LibriSpeech/dev_tsv_file.tsv"
# 수정된 .tsv 파일 경로
output_tsv_path = "/data1/hslim/PycharmProjects/hubert/DB/LibriSpeech/dev_tsv_file_with_nsample.tsv"

# 루트 디렉토리
with open(input_tsv_path, "r") as infile, open(output_tsv_path, "w") as outfile:
    lines = infile.readlines()

    # 첫 줄은 루트 디렉토리
    root_dir = lines[0].strip()
    outfile.write(root_dir + "\n")

    # 나머지 줄은 상대 경로
    for line in lines[1:]:
        relative_path = line.strip()
        full_path = os.path.join(root_dir, relative_path)

        # 오디오 파일인지 확인
        if os.path.isfile(full_path) and full_path.endswith(".wav"):
            try:
                # 샘플 수 계산
                audio, sample_rate = sf.read(full_path)
                nsample = len(audio)
                outfile.write(f"{relative_path}\t{nsample}\n")
            except Exception as e:
                print(f"Error processing {full_path}: {e}")
        else:
            print(f"Invalid file or not a .wav file: {full_path}")
