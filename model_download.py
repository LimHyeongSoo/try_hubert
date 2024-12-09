from transformers import Wav2Vec2Processor, HubertForCTC

# 저장 경로 설정
save_path = "/data1/hslim/PycharmProjects/hubert/models"

# 프로세서 다운로드 및 저장
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
processor.save_pretrained(save_path)

# 모델 다운로드 및 저장
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
model.save_pretrained(save_path)

print(f"Model and processor saved to {save_path}")
