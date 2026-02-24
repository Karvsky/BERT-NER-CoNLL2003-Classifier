import torch.nn as nn
from transformers import AutoModel

class NERModel(nn.Module):
    def __init__(self, num_labels):
        super(NERModel, self).__init__()

        self.bert = AutoModel.from_pretrained("bert-base-cased")

        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0] 
        
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        return logits