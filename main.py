import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from model import NERModel
from dataset_transformation import ConllDataset

def train():
    BATCH_SIZE = 16
    EPOCHS = 1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Uruchamiam na: {DEVICE}")

    train_ds = ConllDataset("train")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = NERModel(num_labels=9).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for epoch in range(EPOCHS):
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(ids, mask)

            loss = loss_fn(outputs.view(-1, 9), labels.view(-1))
            
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), "ner_model.pth")
    print("Gotowe! Model zapisany jako ner_model.pth")

if __name__ == "__main__":
    train()