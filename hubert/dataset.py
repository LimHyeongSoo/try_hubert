# dataset.py
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio

class ASRDataset(Dataset):
    def __init__(
        self,
        root: Path,
        tsv_path: Path,
        ltr_path: Path,
        dict_path: Path,
        sample_rate: int = 16000,
        min_samples: int = 32000,
        max_samples: int = 250000,
        train: bool = True,
    ):
        self.root = root
        self.sample_rate = sample_rate
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.train = train

        # char_dict 로드
        self.char_dict = {}
        with open(dict_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                ch = line.strip()
                self.char_dict[ch] = i

        # tsv 파일 로드
        with open(tsv_path, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
        subset_name = lines[0].strip()
        self.audio_list = []
        for line in lines[1:]:
            if not line.strip():
                continue
            rel_path, nsamples = line.split('\t')
            nsamples = int(nsamples)
            wav_path = self.root / subset_name / rel_path
            if nsamples > self.min_samples:  # 길이 필터링 (옵션)
                self.audio_list.append((wav_path, nsamples))

        # ltr 파일 로드
        self.ltr_dict = {}
        with open(ltr_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('|')
                utt_id = parts[0]
                # 나머지 부분이 word 단위로 구성된 경우, space로 join 후 char 단위 분해
                words = parts[1:]
                transcript = ' '.join(words)  # 공백 포함
                self.ltr_dict[utt_id] = transcript

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, index):
        wav_path, nsamples = self.audio_list[index]
        wav, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        if wav.shape[-1] > self.max_samples:
            diff = wav.shape[-1] - self.max_samples
            offset = random.randint(0, diff)
            wav = wav[:, offset:offset+self.max_samples]

        utt_id = wav_path.stem  # 예: 8629-261140-0005 형태
        transcript = self.ltr_dict[utt_id]
        targets = []
        for ch in transcript:
            if ch in self.char_dict:
                targets.append(self.char_dict[ch])
            # else:
            #   만약 사전에 없는 문자를 어떻게 처리할지 결정 필요. 여기서는 무시.

        targets = torch.tensor(targets, dtype=torch.long)
        return wav, targets

    def collate(self, batch):
        wavs, targets = zip(*batch)
        max_wav_len = max(w.size(-1) for w in wavs)
        coll_wavs = []
        for w in wavs:
            if w.shape[-1] < max_wav_len:
                w = F.pad(w, (0, max_wav_len - w.shape[-1]))
            coll_wavs.append(w)
        coll_wavs = torch.stack(coll_wavs, dim=0)  # (B,1,T)

        max_tgt_len = max(t.size(0) for t in targets)
        coll_targets = torch.full((len(targets), max_tgt_len), fill_value=-1, dtype=torch.long)
        for i, t in enumerate(targets):
            coll_targets[i, :t.size(0)] = t

        return coll_wavs, coll_targets
