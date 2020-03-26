import numpy as np
import os, tqdm, time, json
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset 
from tokenization import FullTokenizer
from Bert import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weights_path = "NN-NLP-Project-Data/uncased_L-12_H-768_A-12/bert_model.ckpt"
vocab_file = "NN-NLP-Project-Data/uncased_L-12_H-768_A-12/vocab.txt"
model_name = "SentenceRetrieval.pt"


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

def load_data(fname):
    f = open(fname, encoding='utf8')
    data = []
    claim_ids = []
    labels = []
    predicted_evidence = []
    for line in f:
        line = json.loads(line)
        sentence = ["[CLS]" + line['claim'] + "[SEP]", line['doc'] + " " + line['sentence'] + "[SEP]"]
        label = line['is_evidence']
        data.append(sentence)
        labels.append(label)
        claim_ids.append(line['id'])
        predicted_evidence.append([line['doc'], line['sid'], line['claim'], line['sentence'], line['label']])
    f.close()
    
    return data, labels, claim_ids, predicted_evidence

def preprocess(data):
    tokenizer = FullTokenizer(vocab_file)
    tok_ip = np.zeros((len(data), 128), dtype="int32")
    sent_ip = np.zeros((len(data), 128), dtype="int8")
    pos_ip = np.zeros((len(data), 128), dtype="int8")
    masks = np.zeros((len(data), 128), dtype="int8")
    
    for pos, text in tqdm.tqdm_notebook(enumerate(data)):
        tok0 = tokenizer.tokenize(text[0])
        tok1 = tokenizer.tokenize(text[1])
        tok = tok0 + tok1
        if len(tok) > 128:
            tok = tok[:127] + ["[SEP]"]
        pad_len = 128-len(tok)
        tok_len = len(tok)
        tok0_len = len(tok0)
        tok = tokenizer.convert_tokens_to_ids(tok) + [0]*pad_len
        pos_val = range(128)
        sent = [0]*tok0_len + [1]*(tok_len-tok0_len) + [0]*pad_len
        mask = [1]*tok_len + [0]*pad_len
        
        tok_ip[pos] = tok
        pos_ip[pos] = pos_val
        masks[pos] = mask
        
    masks = masks[:, None, None, :]
    return tok_ip, sent_ip, pos_ip, masks

data_train, labels_train, ids_train, predicted_evidence_train = load_data("NN-NLP-Project-Data/train-data.jsonl")

if not os.path.exists("train/train-tok.npy"):
    tok_ip, sent_ip, pos_ip, masks = preprocess(data_train)
    labels = np.array(labels_train)
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

data_dev, labels_dev, ids_dev, predicted_evidence_dev = load_data("NN-NLP-Project-Data/dev-data.jsonl")

if not os.path.exists("dev/dev-tok.npy"):
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

data_test, labels_test, ids_test, predicted_evidence_test = load_data("NN-NLP-Project-Data/test-data.jsonl")

if not os.path.exists("test/test-tok.npy"):
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

def train(model, loader, criterion, optimizer):
    model.train()
    loss_epoch = 0
    idx = 0
    for tok_ip, sent_ip, pos_ip, masks, y in tqdm.tqdm(loader):
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
        idx += 1
        if idx % 500 == 0:
            print("Loss:", loss_epoch/idx)
            torch.save(model.state_dict(), model_name)
    print ("Loss:", loss_epoch/len(loader))
    
    return loss_epoch/len(loader)

def test(model, loader):
    model.eval()
    outputs = []
    scores = []
    for tok_ip, sent_ip, pos_ip, masks, y in tqdm.tqdm(loader):
        optimizer.zero_grad()
        tok_ip = tok_ip.type(torch.LongTensor).to(device)
        sent_ip = sent_ip.type(torch.LongTensor).to(device)
        pos_ip = pos_ip.type(torch.LongTensor).to(device)
        masks = masks.type(torch.FloatTensor).to(device)
        y = y.to(device)
        output = model(tok_ip, sent_ip, pos_ip, masks)
        
        scores.extend(output.detach().cpu().numpy()[:, 1])
        outputs.extend(output.detach().cpu().argmax(dim=1).numpy())
       
    return np.asarray(outputs), np.asarray(scores)

# Get top 5 evidences for each claim
def get_top_5(preds, scores, ids, predicted_evidence):
    
    evidence_map = {}
    top_5_map = {}
    
    for i in range(len(ids)):
        
        # if preds[i] != 1:
        #     continue
        if ids[i] not in evidence_map.keys():
            evidence_map[ids[i]] = []
        evidence_map[ids[i]].append((scores[i], predicted_evidence[i]))
        
    for id, sents in evidence_map.items():
        top_5_sents = sorted(sents, key=lambda x: x[0], reverse=True)[:5]
        top_5_map[id] = top_5_sents
    
    return top_5_map

# Make final json with id, label, predicted_label, evidence and predicted_evidence
def format_output(out_path, top_5_map):
    
    outputs = []
    for id, sents in top_5_map.items():
        
        for sent, meta in sents:
            output_obj = {}
            output_obj['id'] = id
            output_obj['claim'] =  meta[2]
            output_obj['label'] = meta[4]
            output_obj['doc'] = meta[0]
            output_obj['sid'] = meta[1]
            output_obj['sentence'] = meta[3]
            
            outputs.append(output_obj)

    # Write final predictions to file
    with open(out_path, 'w', encoding='utf8') as f:
        for line in outputs:
            json.dump(line, f)
            f.write("\n")


train_dataset = SentenceDataset(tok_ip, sent_ip, pos_ip, masks, labels)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64, num_workers=8)

dev_dataset = SentenceDataset(tok_ip_dev, sent_ip_dev, pos_ip_dev, masks_dev, labels_dev)
dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=32, num_workers=8)


test_dataset = SentenceDataset(tok_ip_test, sent_ip_test, pos_ip_test, masks_test, labels_test)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32, num_workers=8)

config = Config()
model = SentenceRetrieval(config)
load_model(model, weights_path)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# print('Loading model')
# model.load_state_dict(torch.load(model_name))
model.to(device)

for i in range(1):
    x = train(model, dev_loader, criterion, optimizer)
    torch.save(model.state_dict(), model_name)

# Train Set
preds, scores = test(model, train_loader)
top_5_map = get_top_5(preds, scores, ids_train, predicted_evidence_train)
format_output('train_sent_results.txt',top_5_map)

# Dev Set
preds, scores = test(model, dev_loader)
top_5_map = get_top_5(preds, scores, ids_dev, predicted_evidence_dev)
format_output('dev_sent_results.txt', top_5_map)

# Test Set
preds, scores = test(model, test_loader)
top_5_map = get_top_5(preds, ids_test, predicted_evidence_test)
format_output('test_sent_results.txt', top_5_map)

