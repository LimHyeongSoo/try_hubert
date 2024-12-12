import os
from pydub import AudioSegment

# .flac 파일을 .wav로 변환하는 함수
def convert_flac_to_wav(root_dir):
    try:
        file_count = 0  # 변환된 파일 수를 세기 위한 변수
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".flac"):
                    flac_path = os.path.join(root, file)
                    wav_path = os.path.splitext(flac_path)[0] + ".wav"

                    # .flac 파일을 .wav로 변환
                    audio = AudioSegment.from_file(flac_path, format="flac")
                    audio.export(wav_path, format="wav")
                    file_count += 1
                    print(f"Converted: {flac_path} -> {wav_path}")

        print(f"모든 변환 작업이 완료되었습니다. 총 {file_count}개의 파일을 변환했습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

# 변환할 최상위 디렉토리 설정
root_directory = "/data3/hslim/PycharmProjects/try_hubert/DB/LibriSpeech/train-clean-100"
convert_flac_to_wav(root_directory)
