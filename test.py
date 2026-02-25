import torch
from torch.utils.data import DataLoader
from dataset_transformation import ConllDataset
from model import NERModel 
from seqeval.metrics import classification_report
import numpy as np
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"

def evaluate():
    try:
        test_ds = ConllDataset("valid") 
        test_loader = DataLoader(test_ds, batch_size=8)
    except:
        return

    model = NERModel(num_labels=9)
    model_path = "ner_model.pth" if os.path.exists("ner_model.pth") else "model.safetensors"
    
    if not os.path.exists(model_path):
        return

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    id2tag = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 
              5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"}

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = model(input_ids, attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            preds = torch.argmax(logits, dim=2)

            for i in range(labels.shape[0]):
                temp_preds = []
                temp_labels = []
                for j in range(labels.shape[1]):
                    if labels[i][j] != -100:
                        temp_preds.append(id2tag[preds[i][j].item()])
                        temp_labels.append(id2tag[labels[i][j].item()])
                
                if temp_labels:
                    all_preds.append(temp_preds)
                    all_labels.append(temp_labels)

    print("\n" + "="*50)
    print("           MODEL EVALUATION REPORT")
    print("="*50)
    print(classification_report(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()