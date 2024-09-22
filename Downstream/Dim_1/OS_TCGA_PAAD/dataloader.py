import os
import torch
import torch.utils.data as data
from transformers import BertTokenizer

class TCGA_PAAD_Dataset(data.Dataset):
    def __init__(self, data_path, split="train", transform=None, max_words=2048):
        super().__init__()
        if not os.path.exists(data_path):
            raise RuntimeError(f"{data_path} does not exist!")

        self.transform = transform
        self.data_path = data_path
        self.samples = self.prepare_raw_data(os.path.join(data_path, split+".csv"))  # Assuming CSV format for survival data

        self.tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.max_words = max_words

        print(split, "dataset samples:", self.__len__())

    def prepare_raw_data(self, filepath):
        import pandas as pd
        df = pd.read_csv(filepath)

        # Assuming columns: ['text', 'survival_time', 'censor']
        samples = []
        for index, row in df.iterrows():
            sample = {}
            sample["text"] = row["report_text"]
            sample["survival_time"] = row["survival_time"]
            sample["censor"] = row["censor"]
            samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def get_embedding(self, sent):
        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        return tokens

    def __getitem__(self, index):
        item = self.samples[index]
        text = item["text"]
        censor = torch.tensor([item["censor"]], dtype=torch.float32)
        survival_time = torch.tensor([item["survival_time"]], dtype=torch.float32)

        tokens = self.get_embedding(text)
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        return input_ids, survival_time, censor, attention_mask
