import utils
from classifier import TextClassifier
from dataset import TextDataset

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW 
from transformers import BertTokenizer, BertModel

# pickle to list 
mosi_train = pd.read_pickle('data/MOSI/train.pkl')
mosi_dev = pd.read_pickle('data/MOSI/dev.pkl')
mosi_test = pd.read_pickle('data/MOSI/test.pkl')

mosei_train = pd.read_pickle('data/MOSEI/train.pkl')
mosei_dev = pd.read_pickle('data/MOSEI/dev.pkl')
mosei_test = pd.read_pickle('data/MOSEI/test.pkl')

# modality division
train_ms_txt = utils.modality_div(mosi_train, 0)
train_ms_aud = utils.modality_div(mosi_train, 2)
train_ms_labels = utils.modality_div(mosi_train, -1)

dev_ms_txt = utils.modality_div(mosi_dev, 0)
dev_ms_aud = utils.modality_div(mosi_dev, 2)
dev_ms_labels = utils.modality_div(mosi_dev, -1)

test_ms_txt = utils.modality_div(mosi_test, 0)
test_ms_aud = utils.modality_div(mosi_test, 2)
test_ms_labels = utils.modality_div(mosi_test, -1)

# tokenization and classification
train_ms_txt = utils.tokenized_data(train_ms_txt)
train_ms_lab = utils.reg2cls(train_ms_labels)

dev_ms_txt = utils.tokenized_data(dev_ms_txt)
dev_ms_lab = utils.reg2cls(dev_ms_labels)

test_ms_txt = utils.tokenized_data(test_ms_txt)
test_ms_lab = utils.reg2cls(test_ms_labels)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 인코딩된 문장들을 텍스트 형식으로 변환
train_decoded_txt = [tokenizer.decode(txt, skip_special_tokens=True) for txt in train_ms_txt]
dev_decoded_txt = [tokenizer.decode(txt, skip_special_tokens=True) for txt in dev_ms_txt]
test_decoded_txt = [tokenizer.decode(txt, skip_special_tokens=True) for txt in test_ms_txt]

# Tokenize 문장들, 패딩 및 어텐션 마스크 생성
train_encoded_inputs = tokenizer(train_decoded_txt, padding = "longest", return_tensors = "pt")
train_input_ids = train_encoded_inputs['input_ids']
train_attention_mask = train_encoded_inputs["attention_mask"]

dev_encoded_inputs = tokenizer(dev_decoded_txt, padding = "longest", return_tensors = "pt")
dev_input_ids = dev_encoded_inputs['input_ids']
dev_attention_mask = dev_encoded_inputs["attention_mask"]

test_encoded_inputs = tokenizer(test_decoded_txt, padding = "longest", return_tensors = "pt")
test_input_ids = test_encoded_inputs['input_ids']
test_attention_mask = test_encoded_inputs["attention_mask"]

# TensorDataset으로 변환
train_dataset = TextDataset(train_input_ids, train_attention_mask, train_ms_lab)
dev_dataset = TextDataset(dev_input_ids, dev_attention_mask, dev_ms_lab)
test_dataset = TextDataset(test_input_ids, test_attention_mask, test_ms_lab)

# DataLoader 사용
train_dataloader = DataLoader(train_dataset, batch_size = 16)
dev_dataloader = DataLoader(dev_dataset, batch_size = 16)
test_dataloader = DataLoader(test_dataset, batch_size = 16)
num_cls = 3

# BERT 모델 정의
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

# 모델 분류기 및 준비
txt_model = TextClassifier(bert_model, num_cls)
optimizer = AdamW(txt_model.parameters(), lr = 5e-5)
criterion = nn.CrossEntropyLoss()

# 학습
txt_model.to(device)

num_epochs = 5

for epoch in range(num_epochs):
    txt_model.train()
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device).long()

        optimizer.zero_grad()
        logits = txt_model(input_ids, attention_mask = attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}: Loss = {loss.item()}")
    
# 모델 평가
txt_model.eval()
total_acc = 0
num_batches = 0

# 그래디언트 계산 비활성화
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = txt_model(input_ids, attention_mask=attention_mask)
        acc = utils.accuracy(logits, labels)

        total_acc += acc
        num_batches += 1

# 모델 성능 평가
test_accuracy = total_acc / num_batches
print(f"Test Accuracy: {test_accuracy:.2f}")