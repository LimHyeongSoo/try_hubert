import torchaudio
import torch
from torch.utils.data import Dataset
from pathlib import Path


class ASRDataset(Dataset):
    def __init__(self, root: Path, tsv_path: Path, ltr_path: Path, tokenizer, train=True):
        """
        root: 데이터 루트 디렉토리
        tsv_path: 오디오 리스트가 들어있는 TSV 파일
                  형식:
                  첫 줄: 오디오 루트 디렉토리
                  이후: 파일경로 \t duration
        ltr_path: 전사 정보 파일 (file_id|word1|word2|...)
        tokenizer: 토크나이저
        train: True/False
        """
        self.root = root
        self.tokenizer = tokenizer
        self.train = train

        # TSV 로딩
        with open(tsv_path, "r") as f:
            lines = f.readlines()
        self.audio_root = Path(lines[0].strip())

        # 두 번째 줄부터: "상대경로\t길이" 형태이므로 split 필요
        self.audio_paths = []
        for line in lines[1:]:
            line = line.strip()
            # 예: "8629/261140/8629-261140-0024.wav    199600"
            parts = line.split()  # 공백 기준 split
            rel_path = parts[0]  # 첫 번째 필드는 오디오 상대 경로
            # duration = parts[1] # 필요하다면 여기서 사용할 수 있음, 현재는 불필요
            self.audio_paths.append(self.audio_root / rel_path)

        # LTR 로딩
        self.ltr_dict = {}
        with open(ltr_path, "r") as f:
            for line in f:
                line_parts = line.strip().split("|")
                file_id = line_parts[0]
                words = line_parts[1:]
                transcript_str = " ".join(words)
                self.ltr_dict[file_id] = transcript_str

        # audio_paths를 바탕으로 매칭
        self.items = []
        for ap in self.audio_paths:
            file_id = ap.stem
            if file_id in self.ltr_dict:
                self.items.append((ap, self.ltr_dict[file_id]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, transcript_str = self.items[idx]
        wav, sr = torchaudio.load(wav_path)

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000

        target_encoding = self.tokenizer(
            text_target=transcript_str,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False
        )
        target_ids = target_encoding.input_ids[0]

        return wav, target_ids, transcript_str

    def collate(self, batch):
        # batch: list of (wav, target_ids, transcript_str)
        wavs = [b[0].squeeze(0) for b in batch]  # [C,T], C=1 가정 -> [T]
        targets = [b[1] for b in batch]
        transcripts = [b[2] for b in batch]

        # wav padding
        wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)

        # target padding: blank=0 가정
        max_len = max(len(t) for t in targets)
        padded_targets = []
        for t in targets:
            padded = torch.cat([t, torch.full((max_len - len(t),), 0, dtype=torch.long)])
            padded_targets.append(padded)
        padded_targets = torch.stack(padded_targets, dim=0)

        return wavs, padded_targets, transcripts
