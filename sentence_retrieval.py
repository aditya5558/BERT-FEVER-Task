#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os, tqdm, time, json
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset 
from multiprocessing.dummy import Pool as ThreadPool


# In[2]:


from tokenization import FullTokenizer
from Bert import *


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[4]:


weights_path = "uncased_L-12_H-768_A-12/bert_model.ckpt"
vocab_file = "uncased_L-12_H-768_A-12/vocab.txt"
model_name = "SentenceRetrieval"


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
        
        self.output = nn.Linear(config.emb_dim, 2)
        
    def forward(self, token_ip, sent_ip, pos_ip, mask=None):
        embeddings = self.enbedding_layer(token_ip, sent_ip, pos_ip)
        for encoder in self.encoders:
            embeddings = encoder(embeddings, mask)
        out = self.output(embeddings[:, 0])
        
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


# In[8]:


def preprocess(data):
    tokenizer = FullTokenizer(vocab_file)
    tok_ip = np.zeros((len(data), 512), dtype="int32")
    sent_ip = np.zeros((len(data), 512), dtype="int8")
    pos_ip = np.zeros((len(data), 512), dtype="int8")
    masks = np.zeros((len(data), 512), dtype="int8")
    
    for pos, text in tqdm.tqdm_notebook(enumerate(data)):
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
        sent_ip[pos] = sent
        
    masks = masks[:, None, None, :]
    return tok_ip, sent_ip, pos_ip, masks


# In[9]:


if not os.path.exists("train/train-tok.npy"):
    data, labels = load_data("train-data.jsonl")
    
#     pool = ThreadPool(16)
#     tok_ip, sent_ip, pos_ip, masks = list(tqdm.tqdm(pool.imap(preprocess, data), total=len(data))
    
    tok_ip, sent_ip, pos_ip, masks = preprocess(data)
    labels = np.array(labels)
    os.mkdir("train")
    np.save("train/train-tok.npy", tok_ip)
    np.save("train/train-sent.npy", sent_ip)
    np.save("train/train-pos.npy", pos_ip)
    np.save("train/train-masks.npy", masks)
    np.save("train/train-labels.npy", labels)
else:
    tok_ip = np.load("train/train-tok.npy")
    sent_ip = np.load("train/train-sent.npy")
    pos_ip = np.load("train/train-pos.npy")
    masks = np.load("train/train-masks.npy")
    labels = np.load("train/train-labels.npy")   


# In[10]:


if not os.path.exists("dev/dev-tok.npy"):
    data_dev, labels_dev = load_data("dev-data.jsonl")
    tok_ip_dev, sent_ip_dev, pos_ip_dev, masks_dev = preprocess(data_dev)
    labels_dev = np.array(labels_dev)
    os.mkdir("dev")
    np.save("dev/dev-tok.npy", tok_ip_dev)
    np.save("dev/dev-sent.npy", sent_ip_dev)
    np.save("dev/dev-pos.npy", pos_ip_dev)
    np.save("dev/dev-masks.npy", masks_dev)
    np.save("dev/dev-labels.npy", labels_dev)
else:
    tok_ip_dev = np.load("dev/dev-tok.npy")
    sent_ip_dev = np.load("dev/dev-sent.npy")
    pos_ip_dev = np.load("dev/dev-pos.npy")
    masks_dev = np.load("dev/dev-masks.npy")
    labels_dev = np.load("dev/dev-labels.npy")


# In[ ]:


if not os.path.exists("test/test-tok.npy"):
    data_test, labels_test = load_data("test-data.jsonl")
    tok_ip_test, sent_ip_test, pos_ip_test, masks_test = preprocess(data_test)
    labels_test = np.array(labels_test)
    os.mkdir("test")
    np.save("test/test-tok.npy", tok_ip_test)
    np.save("test/test-sent.npy", sent_ip_test)
    np.save("test/test-pos.npy", pos_ip_test)
    np.save("test/test-masks.npy", masks_test)
    np.save("test/test-labels.npy", labels_test)
else:
    tok_ip_test = np.load("test/test-tok.npy")
    sent_ip_test = np.load("test/test-sent.npy")
    pos_ip_test = np.load("test/test-pos.npy")
    masks_test = np.load("test/test-masks.npy")
    labels_test = np.load("test/test-labels.npy")


# In[ ]:


def train(model, loader, criterion, optimizer):
    model.train()
    loss_epoch = 0
    for tok_ip, sent_ip, pos_ip, masks, y in tqdm.tqdm_notebook(loader):
        optimizer.zero_grad()
        tok_ip = tok_ip.type(torch.LongTensor).to(device)
        sent_ip = sent_ip.type(torch.LongTensor).to(device)
        pos_ip = pos_ip.type(torch.LongTensor).to(device)
        masks = masks.type(torch.FloatTensor).to(device)
        y = y.to(device)
        O = model(tok_ip, sent_ip, pos_ip, masks)
        loss = criterion(O, y)
        loss_epoch += loss.item()
        loss.backward()
        optimizer.step()
    print ("Loss:", loss_epoch/len(loader))
    
    return loss_epoch/len(loader)


# In[ ]:


train_dataset = SentenceDataset(tok_ip, sent_ip, pos_ip, masks, labels)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16, num_workers=4)


# In[ ]:


dev_dataset = SentenceDataset(tok_ip_dev, sent_ip_dev, pos_ip_dev, masks_dev, labels_dev)
dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=16, num_workers=4)


# In[ ]:


test_dataset = SentenceDataset(tok_ip_test, sent_ip_test, pos_ip_test, masks_test, labels_test)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=16, num_workers=4)


# In[ ]:


config = Config()
model = SentenceRetrieval(config)
load_model(model, weights_path)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model.to(device)


# In[ ]:


for i in range(1):
    x = train(model, dev_loader, criterion, optimizer)
    torch.save(model.state_dict(), model_name)


# In[ ]:




