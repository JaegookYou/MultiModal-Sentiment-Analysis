import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, hidden_size, bert_model, num_classes):
        super().__init__()
        self.bert_model = bert_model
        self.dropout = nn.Dropout(0.1)
        self.conv_layer = nn.Conv1d(in_channels = hidden_size, out_channels = 1, kernel_size = 1)
        self.linear_layer = nn.Linear(1, num_classes)

    def forward(self, input_id, att_mask):
        bert_outputs = self.bert_model(input_ids = input_id, attention_mask = att_mask)[0]
        batch_size, seq_length, hidden_size = bert_outputs.shape
        bert_output_2d = bert_outputs.view(-1, hidden_size).unsqueeze(1)
        conv_output = self.conv_layer(bert_output_2d.transpose(2, 1)).squeeze()
        linear_output = self.linear_layer(conv_output)
        return linear_output
    