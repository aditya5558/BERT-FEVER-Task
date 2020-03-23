
# coding: utf-8

# In[1]:


import numpy as np
import os, tqdm, time, json
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset 


# In[2]:


from tokenization import FullTokenizer
from Bert import *


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[4]:


weights_path = "uncased_L-12_H-768_A-12/bert_model.ckpt"
vocab_file = "uncased_L-12_H-768_A-12/vocab.txt"


# In[5]:


class SentenceDataset(Dataset):
    def __init__(self, tok_ip, sent_ip, pos_ip, masks, y):
        self.tok_ip = tok_ip
        self.sent_ip = sent_ip
        self.pos_ip = pos_ip
        self.masks = masks
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.tok_ip[index], self.sent_ip[index], self.pos_ip[index], self.masks[index], self.y[index]


# In[6]:


class SentenceRetrieval(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enbedding_layer = EmbeddingLayer(config)
        self.encoders = nn.ModuleList([EncoderLayer(config) for i in range(config.num_encoders)])
        
        self.linear = nn.Linear(config.emb_dim, config.emb_dim)
        self.drp = nn.Dropout(config.fc_drop_rate)
        self.activation = GELU()
        self.output = nn.Linear(config.emb_dim, 2)
        
    def forward(self, token_ip, sent_ip, pos_ip, mask=None):
        embeddings = self.enbedding_layer(token_ip, sent_ip, pos_ip)
        for encoder in self.encoders:
            embeddings = encoder(embeddings, mask)
        lin_out = self.activation(self.drp(self.linear(embeddings[:, 0])))
        out = self.output(lin_out)
        
        return out


# In[7]:


def load_data(fname):
    f = open(fname, encoding='utf8')
    data = []
    labels = []
    for line in f:
        line = json.loads(line)
        sentence = ["[CLS]" + line['claim'] + "[SEP]", line['sentence'] + "[SEP]"]
        label = line['is_evidence']
        data.append(sentence)
        labels.append(label)
    f.close()
    
    return data, labels


# In[34]:


def preprocess(data):
    tokenizer = FullTokenizer(vocab_file)
    tok_ip = np.zeros((len(data), 512), dtype="long")
    sent_ip = np.zeros((len(data), 512), dtype="long")
    pos_ip = np.zeros((len(data), 512), dtype="long")
    masks = np.zeros((len(data), 512), dtype="float32")
    
    for pos, text in tqdm.tqdm_notebook(enumerate(data), total=len(data)):
        tok0 = tokenizer.tokenize(text[0])
        tok1 = tokenizer.tokenize(text[1])
        tok = tok0 + tok1
        if len(tok) > 512:
            tok = tok[:511] + ["[SEP]"]
        pad_len = 512-len(tok)
        tok_len = len(tok)
        tok0_len = len(tok0)
        tok = tokenizer.convert_tokens_to_ids(tok) + [0]*pad_len
        pos_val = range(512)
        sent = [0]*tok0_len + [1]*(tok_len-tok0_len) + [0]*pad_len
        mask = [1]*tok_len + [0]*pad_len
        
        tok_ip[pos] = tok
        pos_ip[pos] = pos_val
        masks[pos] = mask
        
    masks = masks[:, None, None, :]
    return tok_ip, sent_ip, pos_ip, masks


# In[35]:


data, labels = load_data("dev-data.jsonl")


# In[36]:


tok_ip, sent_ip, pos_ip, masks = preprocess(data)
labels = np.array(labels)


# In[37]:


def train(model, loader, criterion, optimizer):
    model.train()
    loss_epoch = 0
    for tok_ip, sent_ip, pos_ip, masks, y in tqdm.tqdm_notebook(loader):
        optimizer.zero_grad()
        tok_ip = tok_ip.to(device)
        sent_ip = sent_ip.to(device)
        pos_ip = pos_ip.to(device)
        masks = masks.to(device)
        y = y.to(device)
        O = model(tok_ip, sent_ip, pos_ip, masks)
        loss = criterion(O, y)
        loss_epoch += loss.item()
        loss.backward()
        optimizer.step()
    print ("Loss:", loss_epoch/len(loader))
    
    return loss_epoch/len(loader)


# In[38]:


train_dataset = SentenceDataset(tok_ip, sent_ip, pos_ip, masks, labels)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16, num_workers=4)


# In[ ]:


config = Config()
model = SentenceRetrieval(config)
load_model(model, weights_path)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model.to(device)


# In[ ]:


for i in range(10):
    x = train(model, train_loader, criterion, optimizer)

