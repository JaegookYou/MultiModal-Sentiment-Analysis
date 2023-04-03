import preprocessing as pp
import classifier as clsr

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim
from transformers import BertTokenizer, BertModel

# pickle to list 
mosi_train = pd.read_pickle('data/MOSI/train.pkl')
mosi_dev = pd.read_pickle('data/MOSI/dev.pkl')
mosi_test = pd.read_pickle('data/MOSI/dev.pkl')

mosei_train = pd.read_pickle('data/MOSEI/train.pkl')
mosei_dev = pd.read_pickle('data/MOSEI/dev.pkl')
mosei_test = pd.read_pickle('data/MOSEI/dev.pkl')

# modality division
train_ms_txt = pp.modality_div(mosi_train, 0)
train_ms_aud = pp.modality_div(mosi_train, 2)
train_ms_labels = pp.modality_div(mosi_train, -1)

dev_ms_txt = pp.modality_div(mosi_dev, 0)
dev_ms_aud = pp.modality_div(mosi_dev, 2)
dev_ms_labels = pp.modality_div(mosi_dev, -1)

# tokenization and classification
train_ms_txt = pp.tokenized_data(train_ms_txt)
train_ms_lab = pp.reg2cls(train_ms_labels)

dev_ms_txt = pp.tokenized_data(dev_ms_txt)
dev_ms_lab = pp.reg2cls(dev_ms_labels)
num_cls = 3

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 인코딩된 문장들을 텍스트 형식으로 변환
train_decoded_txt = [tokenizer.decode(txt, skip_special_tokens=True) for txt in train_ms_txt]
dev_decoded_txt = [tokenizer.decode(txt, skip_special_tokens=True) for txt in dev_ms_txt]

# Tokenize 문장들, 패딩 및 어텐션 마스크 생성
train_encoded_inputs = tokenizer(train_decoded_txt, padding = "longest", return_tensors = "pt")
train_input_ids = train_encoded_inputs['input_ids']
train_attention_mask = train_encoded_inputs["attention_mask"]

dev_encoded_inputs = tokenizer(dev_decoded_txt, padding = "longest", return_tensors = "pt")
dev_input_ids = dev_encoded_inputs['input_ids']
dev_attention_mask = dev_encoded_inputs["attention_mask"]

# TensorDataset으로 변환
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_ms_lab)
dev_dataset = TensorDataset(dev_input_ids, dev_attention_mask, dev_ms_lab)

# DataLoader 사용
train_dataloader = DataLoader(train_dataset, batch_size = 32)
dev_dataloader = DataLoader(dev_dataset, batch_size = 32)

# 모델 및 분류기 준비
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
txt_model = BertModel.from_pretrained('bert-base-uncased').to(device)

criterion = nn.CrossEntropyLoss()