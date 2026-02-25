import torch
from transformers import AutoTokenizer
from model import NERModel

def test_my_bert():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = NERModel(num_labels=9)
    model.load_state_dict(torch.load("ner_model.pth", map_location=torch.device('cpu')))
    model.eval()

    id2tag = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 
              5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"}

    print("=== BERT NER TESTER ===")
    print("Enter a sentence in English (recommended) to test the model.")
    
    while True:
        sentence = input("\nYour sentence (or 'exit'): ")
        if sentence.lower() == 'exit': break

        inputs = tokenizer(sentence, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            predictions = torch.argmax(logits, dim=2)

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        for token, pred in zip(tokens, predictions[0]):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                tag = id2tag[pred.item()]
                if tag != "O":
                    print(f"{token:15} -> {tag}")

if __name__ == "__main__":
    test_my_bert()