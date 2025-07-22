import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import torch.nn as nn

class TranslationCollator:
    def __init__(self, tokenizer, max_length=None, with_text=True):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, samples):
        src_texts = [s['src_text'] for s in samples]
        tgt_texts = [s['tgt_text'] for s in samples]

        # 소스 문장 인코딩
        src_encoding = self.tokenizer(
            src_texts,
            padding=True,  # 미니배치 내에서 가장 긴 문장 기준으로 패딩
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length if self.max_length else None,
            add_special_tokens=False
        )

        # eos
        tgt_encoding = self.tokenizer(
            tgt_texts,
            padding=True,  
            truncation=True,
            return_tensors="pt",
            add_special_tokens = False,
            max_length=self.max_length if self.max_length else None
        )

        # bos
        label_encoding = self.tokenizer(
            tgt_texts,
            padding=True,  
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length if self.max_length else None,
            add_special_tokens=True
        )


        if self.tokenizer_tgt.bos_token is None:
            self.tokenizer_tgt.add_special_tokens({'bos_token': '<BOS>'}) 

          # BOS 토큰 추가 (맨 앞에 BOS 토큰을 추가)
        bos_token_id = self.tokenizer_tgt.bos_token_id

        bos_token_tensor = torch.full((tgt_encoding['input_ids'].shape[0], 1), bos_token_id, dtype=torch.long)
        tgt_encoding['input_ids'] = torch.cat([bos_token_tensor, tgt_encoding['input_ids']], dim=1)

        return_value = {
            'input_ids': src_encoding['input_ids'],  # 소스 문장 입력
            'attention_mask_src': src_encoding['attention_mask'],  # 패딩 마스킹
            'tgt_ids': tgt_encoding['input_ids'], # 디코더 입력 문장
            'labels': label_encoding['input_ids'],  # 번역된 문장 (목표 문장)
        }

        return return_value

        #  self.with_text = with_text
        # if self.with_text:
        #     return_value['src_text'] = src_texts
        #     return_value['tgt_text'] = tgt_texts


class TranslationDataset(Dataset):
    def __init__(self, split='train'):
        self.dataset = load_dataset("bentrevett/multi30k", split=split)
  
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            'src_text': str(self.dataset[idx]['en']),  # 원본 문장
            'tgt_text': str(self.dataset[idx]['de']),  # 번역 문장
        }







# ------------------------------------------------------------
# 토크나이저 로드 (영어 -> 독일어 번역 예시)
# tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
# # # # 데이터셋 및 데이터로더 생성
# dataset = TranslationDataset()
# collate_fn = TranslationCollator(tokenizer)
# dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
# # 데이터 확인
# batch = next(iter(dataloader))
# print(batch)

# print("b크기",b.size())
# input = len(tokenizer_src) + 1
# em = nn.Embedding(input, 512)
# print("임베딩 후 크기",em(b).size())
# print([tokenizer.decode(label, skip_special_tokens=True) for label in batch['input_ids']])
# print([tokenizer.decode(label, skip_special_tokens=True) for label in batch['tgt_ids']])
# print(tokenizer.decode(3))


