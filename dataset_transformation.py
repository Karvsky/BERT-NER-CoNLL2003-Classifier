import torch
import os
import kagglehub
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

class ConllDataset(Dataset):
    def __init__(self, split_name="train"):
        try:
            print(f"[*] Przygotowuję dane dla: {split_name}")
            base_path = kagglehub.dataset_download("juliangarratt/conll2003-dataset")

            kaggle_map = {
                "train": "eng.train",
                "valid": "eng.testa",
                "test": "eng.testb"
            }
            target_name = kaggle_map.get(split_name, "eng.train")

            file_path = None
            for root, dirs, files in os.walk(base_path):
                if target_name in files:
                    file_path = os.path.join(root, target_name)
                    break
            
            if not file_path:
                raise FileNotFoundError(f"Nie znaleziono pliku {target_name} w {base_path}")

            print(f"[V] Wczytuję plik: {file_path}")
            self.raw_data = self._parse_conll(file_path)
            print(f"[V] Sukces! Załadowano {len(self.raw_data)} zdań.")
            
        except Exception as e:
            print(f"[X] Błąd: {e}")
            raise e

    def _parse_conll(self, file_path):
        data = []
        tag_map = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, 
                   "B-LOC": 5, "I-LOC": 6, "B-MISC": 7, "I-MISC": 8}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            words, labels = [], []
            for line in f:
                line = line.strip()
                if not line or line.startswith("-DOCSTART-"):
                    if words:
                        data.append({"tokens": words, "ner_tags": labels})
                        words, labels = [], []
                    continue
                
                parts = line.split()
                if len(parts) >= 4:
                    words.append(parts[0])
                    tag = parts[3]
                    labels.append(tag_map.get(tag, 0))
        return data

    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        example = self.raw_data[idx]
        tokenized_input = tokenizer(
            example["tokens"], is_split_into_words=True, truncation=True, 
            padding='max_length', max_length=64, return_tensors="pt"
        )
        
        word_ids = tokenized_input.word_ids() 
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100) 
            elif word_idx != previous_word_idx:
                aligned_labels.append(example["ner_tags"][word_idx]) 
            else:
                aligned_labels.append(-100) 
            previous_word_idx = word_idx
            
        return {
            "input_ids": tokenized_input["input_ids"].squeeze(0), 
            "attention_mask": tokenized_input["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels)
        }