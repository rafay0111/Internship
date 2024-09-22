import os
import pandas as pd
import torch
import torch.utils.data as data
from transformers import BertTokenizer

class TCGA_PAAD_Dataset(data.Dataset):
    def __init__(self, data_path, split="train", transform=None, max_words=112):
        super().__init__()
        self.transform = transform
        self.data_path = data_path
        self.max_words = max_words

        # Load the data
        self.data = pd.read_csv(os.path.join(data_path, f'TCGA_PAAD_{split}.csv'))

        self.tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        print(f"{split} dataset samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def get_embedding(self, text):
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])
        return tokens, x_len

    def __getitem__(self, index):
        item = self.data.iloc[index]
        label = torch.Tensor([item["vital_status"]])
        caps, cap_len = self.get_embedding(item["report_text"])
        return caps["input_ids"].squeeze(0), label, caps["attention_mask"].squeeze(0)