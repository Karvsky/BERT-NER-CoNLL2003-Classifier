import torch
from torch.utils.data import DataLoader
from dataset_transformation import ConllDataset
from model import NERModel 
from seqeval.metrics import classification_report
import numpy as np
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"

def evaluate():
    print("[*] Wczytuję dane testowe (zbiór walidacyjny)...")
    try:
        test_ds = ConllDataset("valid") 
        test_loader = DataLoader(test_ds, batch_size=8)
    except Exception as e:
        print(f"[X] Błąd ładowania danych: {e}")
        return

    model = NERModel(num_labels=9)

    model_path = "ner_model.pth" 
    
    if not os.path.exists(model_path):
        if os.path.exists("model.safetensors"):
            model_path = "model.safetensors"
        else:
            print(f"[X] BŁĄD: Nie znaleziono pliku wag modelu (ner_model.pth ani model.safetensors)!")
            return

    print(f"[*] Wczytuję wagi z: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    id2tag = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 
              5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"}

    all_preds = []
    all_labels = []

    print("[*] Rozpoczynam przewidywanie na danych testowych...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = model(input_ids, attention_mask)

            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

            preds = torch.argmax(logits, dim=2)

            for i in range(labels.shape[0]):
                temp_preds = []
                temp_labels = []
                for j in range(labels.shape[1]):
                    if labels[i][j] != -100:
                        pred_tag = id2tag[preds[i][j].item()]
                        true_tag = id2tag[labels[i][j].item()]
                        temp_preds.append(pred_tag)
                        temp_labels.append(true_tag)
                
                if temp_labels:
                    all_preds.append(temp_preds)
                    all_labels.append(temp_labels)

    print("\n" + "="*50)
    print("           RAPORT SKUTECZNOŚCI MODELU")
    print("="*50)
    
    try:
        print(classification_report(all_labels, all_preds))
    except Exception as e:
        print(f"[!] Nie udało się wygenerować raportu: {e}")
        print("Model zakończył testowanie, ale wystąpił problem z formatowaniem tabeli.")

if __name__ == "__main__":
    evaluate()