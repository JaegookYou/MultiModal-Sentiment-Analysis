import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, bert_model, num_labels):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask = attention_mask)
        last_hidden_state = outputs[0]  # BERT의 마지막 은닉 상태
        cls_output = last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits
    