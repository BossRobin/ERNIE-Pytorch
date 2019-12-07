#!/usr/bin/env python
# encoding: utf-8
"""
File Description: 
Author: nghuyong
Mail: nghuyong@163.com
Created Time: 2019-12-07 13:27
"""
import torch
from pytorch_transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('./ERNIE-converted')

# For bert embedding extractor
input_ids = torch.tensor([tokenizer.encode("这是百度的ERNIE1.0模型")])

model = BertModel.from_pretrained('./ERNIE-converted')

all_hidden_states, all_attentions = model(input_ids)[-2:]

print('all_hidden_states shape', all_hidden_states.shape)
print(all_hidden_states)
