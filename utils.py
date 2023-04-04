import numpy as np
import torch

def modality_div(data, mod):
    div_li = []
    
    if mod == 0 or mod == 1 or mod == 2:
        for idx in range(len(data)):
            div_li.append(data[idx][0][mod])
            
    elif mod == -1:
        for idx in range(len(data)):
            div_li.append(data[idx][1][0][0])
        
    else:
        raise Exception("IndexError")
    
    return div_li

def tokenized_data(txt_data):
    cls_token = np.array([101])
    sep_token = np.array([102])
    
    tok_data = txt_data
    
    for idx in range(len(txt_data)):
        tok_data[idx] = np.concatenate((cls_token, tok_data[idx], sep_token), axis=0)
    
    return tok_data
            
def reg2cls(data):
    for i in range(len(data)):
        if data[i] >= 1.0:
            data[i] = 2.0
        elif data[i] < 1.0 and data[i] > -1.0:
            data[i] = 1.0
        else:
            data[i] = 0.0
            
    return data

def accuracy(logits, labels):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total