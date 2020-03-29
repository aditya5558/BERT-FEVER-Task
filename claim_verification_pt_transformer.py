import numpy as np
import os, tqdm, time, json
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

from tokenization import FullTokenizer
from Bert import *
from scorer import fever_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weights_path = "NN-NLP-Project-Data/uncased_L-12_H-768_A-12/bert_model.ckpt"
vocab_file = "NN-NLP-Project-Data/uncased_L-12_H-768_A-12/vocab.txt"
model_name = "ClaimVerification.pt"

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

class ClaimVerification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enbedding_layer = EmbeddingLayer(config)
        self.encoders = nn.ModuleList([nn.TransformerEncoderLayer(d_model=config.emb_dim, nhead=config.num_heads, dim_feedforward=config.hidden_dim, activation="gelu") for i in range(config.num_encoders)])
        self.output = nn.Linear(config.emb_dim, 3)
        
    def forward(self, token_ip, sent_ip, pos_ip, mask=None):
        embeddings = self.enbedding_layer(token_ip, sent_ip, pos_ip)
        embeddings = torch.transpose(embeddings, 0, 1)
        for encoder in self.encoders:
            embeddings = encoder(embeddings, mask)
        out = self.output(embeddings[0, :])
        
        return out

def load_data(fname):
    label_dict = {}
    label_dict['UNK'] = -1
    label_dict['NOT ENOUGH INFO'] = 0
    label_dict['SUPPORTS'] = 1
    label_dict['REFUTES'] = 2
    f = open(fname, encoding='utf8')
    data = []
    claim_ids = []
    labels = []
    predicted_evidence = []
    for line in f:
        line = json.loads(line)
        sentence = ["[CLS]" + line['claim'] + "[SEP]", line['sentence'] + "[SEP]"]
        label = label_dict[line['label']]
        data.append(sentence)
        labels.append(label)
        claim_ids.append(line['id'])
        predicted_evidence.append([line['doc'], line['sid']])
    f.close()
    return data, labels, claim_ids, predicted_evidence

def preprocess(data):
    tokenizer = FullTokenizer(vocab_file)
    tok_ip = np.zeros((len(data), 128), dtype="int32")
    sent_ip = np.zeros((len(data), 128), dtype="int8")
    pos_ip = np.zeros((len(data), 128), dtype="int8")
    masks = np.zeros((len(data), 128), dtype="int8")
    
    for pos, text in tqdm.tqdm(enumerate(data)):
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
        sent_ip[pos] = sent
        
    masks = masks[:, None, None, :]
    return tok_ip, sent_ip, pos_ip, masks

if not os.path.exists("train_claim/train-tok.npy"):
    data, labels, ids, predicted_evidence = load_data("train_sent_results.txt")
    tok_ip, sent_ip, pos_ip, masks = preprocess(data)
    labels = np.array(labels)
    os.mkdir("train_claim")
    np.save("train_claim/train-tok.npy", tok_ip)
    np.save("train_claim/train-sent.npy", sent_ip)
    np.save("train_claim/train-pos.npy", pos_ip)
    np.save("train_claim/train-masks.npy", masks)
    np.save("train_claim/train-labels.npy", labels)
else:
    data, labels, ids, predicted_evidence = load_data("train_sent_results.txt")
    tok_ip = np.load("train_claim/train-tok.npy")
    sent_ip = np.load("train_claim/train-sent.npy")
    pos_ip = np.load("train_claim/train-pos.npy")
    masks = np.load("train_claim/train-masks.npy")
    labels = np.load("train_claim/train-labels.npy")

if not os.path.exists("dev_claim/dev-tok.npy"):
    data_dev, labels_dev, ids_dev, predicted_evidence_dev = load_data("dev_sent_results.txt")
    tok_ip_dev, sent_ip_dev, pos_ip_dev, masks_dev = preprocess(data_dev)
    labels_dev = np.array(labels_dev)
    os.mkdir("dev_claim")
    np.save("dev_claim/dev-tok.npy", tok_ip_dev)
    np.save("dev_claim/dev-sent.npy", sent_ip_dev)
    np.save("dev_claim/dev-pos.npy", pos_ip_dev)
    np.save("dev_claim/dev-masks.npy", masks_dev)
    np.save("dev_claim/dev-labels.npy", labels_dev)
else:
    print('Loading npy files')
    data_dev, labels_dev, ids_dev, predicted_evidence_dev = load_data("dev_sent_results.txt")
    tok_ip_dev = np.load("dev_claim/dev-tok.npy")
    sent_ip_dev = np.load("dev_claim/dev-sent.npy")
    pos_ip_dev = np.load("dev_claim/dev-pos.npy")
    masks_dev = np.load("dev_claim/dev-masks.npy")
    labels_dev = np.load("dev_claim/dev-labels.npy")

if not os.path.exists("test_claim/test-tok.npy"):
    data_test, labels_test, ids_test, predicted_evidence_test = load_data("test_sent_results.txt")
    tok_ip_test, sent_ip_test, pos_ip_test, masks_test = preprocess(data_test)
    labels_test = np.array(labels_test)
    os.mkdir("test_claim")
    np.save("test_claim/test-tok.npy", tok_ip_test)
    np.save("test_claim/test-sent.npy", sent_ip_test)
    np.save("test_claim/test-pos.npy", pos_ip_test)
    np.save("test_claim/test-masks.npy", masks_test)
    np.save("test_claim/test-labels.npy", labels_test)
else:
    data_test, labels_test, ids_test, predicted_evidence_test = load_data("test_sent_results.txt")
    tok_ip_test = np.load("test_claim/test-tok.npy")
    sent_ip_test = np.load("test_claim/test-sent.npy")
    pos_ip_test = np.load("test_claim/test-pos.npy")
    masks_test = np.load("test_claim/test-masks.npy")
    labels_test = np.load("test_claim/test-labels.npy")

masks_train = masks_train.reshape(-1,128)
masks_train = (1-masks_train).astype("bool")

masks_dev = masks_dev.reshape(-1,128)
masks_dev = (1-masks_dev).astype("bool")

masks_test = masks_test.reshape(-1,128)
masks_test = (1-masks_test).astype("bool")


def train(model, loader, criterion, optimizer):
    model.train()
    loss_epoch = 0
    idx = 0
    for tok_ip, sent_ip, pos_ip, masks, y in tqdm.tqdm(loader):
        optimizer.zero_grad()
        tok_ip = tok_ip.type(torch.LongTensor).to(device)
        sent_ip = sent_ip.type(torch.LongTensor).to(device)
        pos_ip = pos_ip.type(torch.LongTensor).to(device)
        masks = masks.type(torch.BoolTensor).to(device)
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

    print("Loss:", loss_epoch/len(loader))
    
    return loss_epoch/len(loader)

def test(model, loader):
    model.eval()
    outputs = []
    for tok_ip, sent_ip, pos_ip, masks, _ in tqdm.tqdm(loader):
        tok_ip = tok_ip.type(torch.LongTensor).to(device)
        sent_ip = sent_ip.type(torch.LongTensor).to(device)
        pos_ip = pos_ip.type(torch.LongTensor).to(device)
        masks = masks.type(torch.BoolTensor).to(device)
        output = model(tok_ip, sent_ip, pos_ip, masks)
        outputs.extend(output.detach().cpu().argmax(dim=1).numpy())

    return np.asarray(outputs)

# Merge predictions for each claim
def merge_preds(preds, ids, predicted_evidence):
    preds_dict = {}
    merged_evidence = []
    cur_id = ids[0]
    # Indices represent NEI, Supports, Refutes
    stats = [0, 0, 0]
    evidence_line = []
    stats[preds[0]] += 1
    for i in range(1,len(ids)):
        if ids[i] == cur_id:
            stats[preds[i]] += 1
            evidence_line.append(predicted_evidence[i])
        else:
            # Label Assignment according to rules mentioned in paper
            if stats[1] > 0:
                preds_dict[cur_id] = ["SUPPORTS", evidence_line]
            elif stats[2] > 0 and stats[1] == 0:
                preds_dict[cur_id] = ["REFUTES", evidence_line]
            elif stats[1] == 0 and stats[2] == 0:
                preds_dict[cur_id] = ["NOT ENOUGH INFO", evidence_line]
            stats = [0, 0, 0]
            cur_id = ids[i]
            stats[preds[i]] += 1
            evidence_line = []
            evidence_line.append(predicted_evidence[i])
    if stats[1] > 0:
        preds_dict[cur_id] = ["SUPPORTS", evidence_line]
    elif stats[2] > 0 and stats[1] == 0:
        preds_dict[cur_id] = ["REFUTES", evidence_line]
    elif stats[1] == 0 and stats[2] == 0:
        preds_dict[cur_id] = ["NOT ENOUGH INFO", evidence_line]
    return preds_dict

# Make final json with id, label, predicted_label, evidence and predicted_evidence
def format_output(in_path, out_path, preds_dict, dev=True):
    outputs = []
    i = 0
    with open(in_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            output_obj = {}
            input_obj = json.loads(line.strip())
            output_obj['id'] = input_obj['id']
            if dev:
                output_obj['label'] = input_obj['label']
            if input_obj['id'] in preds_dict:
                output_obj['predicted_label'] = preds_dict[input_obj['id']][0]
                output_obj['predicted_evidence'] = preds_dict[input_obj['id']][1]
            else:
                output_obj['predicted_label'] = "NOT ENOUGH INFO"
                output_obj['predicted_evidence'] = [['null',0]]
            if dev:
                output_obj['evidence'] = input_obj['evidence']
            outputs.append(output_obj)
            i += 1

    # Calculate Fever score for dev set
    if dev:
        print('Dev Set Results')
        fever_sc, label_accuracy, precision, recall, f1 = fever_score(outputs)
        print('Fever Score: ',fever_sc)     
        print('Label Accuracy: ',label_accuracy)   
        print('Precision: ',precision)       
        print('Recall: ',recall)      
        print('F1 Score: ',f1)    

    # Write final predictions to file
    with open(out_path, 'w', encoding='utf8') as f:
        for line in outputs:
            json.dump(line, f)
            f.write("\n")

train_dataset = SentenceDataset(tok_ip, sent_ip, pos_ip, masks, labels)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32, num_workers=8)

dev_dataset = SentenceDataset(tok_ip_dev, sent_ip_dev, pos_ip_dev, masks_dev, labels_dev)
dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=32, num_workers=8)

test_dataset = SentenceDataset(tok_ip_test, sent_ip_test, pos_ip_test, masks_test, labels_test)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32, num_workers=8)

def load_model(model, checkpoint_file):
    """ Load the pytorch model from checkpoint file """

    # Embedding layer
    e, p = model.enbedding_layer, 'bert/embeddings/'
    load_param(checkpoint_file, {
        e.token_embeddings.weight: p+"word_embeddings",
        e.positional_embeddings.weight: p+"position_embeddings",
        e.sentence_embeddings.weight: p+"token_type_embeddings",
        e.layer_norm.weight:       p+"LayerNorm/gamma",
        e.layer_norm.bias:        p+"LayerNorm/beta"
    })

    # Transformer blocks
    for i in range(len(model.encoders)):
        b, p = model.encoders[i], "bert/encoder/layer_%d/"%i
        load_param(checkpoint_file, {
            b.self_attn.out_proj.weight:          p+"attention/output/dense/kernel",
            b.self_attn.out_proj.bias:            p+"attention/output/dense/bias",
            b.linear1.weight:      p+"intermediate/dense/kernel",
            b.linear1.bias:        p+"intermediate/dense/bias",
            b.linear2.weight:      p+"output/dense/kernel",
            b.linear2.bias:        p+"output/dense/bias",
            b.norm1.weight:          p+"attention/output/LayerNorm/gamma",
            b.norm1.bias:           p+"attention/output/LayerNorm/beta",
            b.norm2.weight:          p+"output/LayerNorm/gamma",
            b.norm2.bias:           p+"output/LayerNorm/beta",
        })
        load_param_num(checkpoint_file, {
            b.self_attn.in_proj_weight:   [p+"attention/self/query/kernel", p+"attention/self/key/kernel", p+"attention/self/value/kernel"],
            b.self_attn.in_proj_bias:     [p+"attention/self/query/bias", p+"attention/self/key/bias", p+"attention/self/value/bias"],
        })

config = Config()
model = ClaimVerification(config)
load_model(model, weights_path)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
model.to(device)

# Train
for i in range(2):
    x = train(model, dev_loader, criterion, optimizer)
    torch.save(model.state_dict(), model_name)

# print('Loading model')
# model.load_state_dict(torch.load(model_name))
# model.to(device)

#Dev Set
preds = test(model, dev_loader)
preds_dict = merge_preds(preds, ids_dev, predicted_evidence_dev)
format_output('dev.jsonl', 'dev_results.txt', preds_dict)

# Test Set
preds = test(model, test_loader)
preds_dict = merge_preds(preds, ids_dev, predicted_evidence_dev, dev=False)
format_output('test.jsonl', 'test_results.txt', preds_dict)
