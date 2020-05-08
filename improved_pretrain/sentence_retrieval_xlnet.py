from transformers import XLNetTokenizer, XLNetForSequenceClassification, AdamW
import torch, tqdm, json
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 1

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased')
model.to(device)

model_name = "SentenceRetrieval.pt"

def process_data(fname, train=True):
    X = []
    y = []
    claim_ids = []
    predicted_evidence = []
    f = open(fname, encoding='utf8')
    f.readline()
    maxi = 0
    for line in f:
        line = json.loads(line)
        claim_ids.append(line['id'])
        predicted_evidence.append([line['doc'], line['sid'], line['claim'], line['sentence'], line['label']])

        input_ids = (tokenizer.encode(line['claim']+" [SEP] "+line['doc'] + " " + line["sentence"], add_special_tokens=True))[:128]
        X.append(input_ids)
        if len(input_ids) > maxi:
            maxi = len(input_ids)
        y.append(line['is_evidence'])
    f.close()
    mask = []
    for val in X:
        mask.append([1]*len(val) + [0]*(maxi-len(val)))
        val += [0]*(maxi-len(val)) #np.pad(val, (0, maxi-len(val)), 'constant')
        
    return torch.LongTensor(X), torch.LongTensor(y), torch.LongTensor(mask), claim_ids, predicted_evidence

X_train, y_train, mask_train, ids_train, predicted_evidence_train = process_data("NN-NLP-Project-Data/train-data.jsonl")
X_dev, y_dev, mask_dev, ids_dev, predicted_evidence_dev = process_data("NN-NLP-Project-Data/dev-data.jsonl")
X_test, y_test, mask_test, ids_test, predicted_evidence_test = process_data("NN-NLP-Project-Data/test-data.jsonl")

train_dataset = TensorDataset(X_train, y_train, mask_train)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32, num_workers=8)

dev_dataset = TensorDataset(X_dev, y_dev, mask_dev)
dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=32, num_workers=8)

test_dataset = TensorDataset(X_test, y_test, mask_test)
test_loader = DataLoader(dev_dataset, shuffle=False, batch_size=32, num_workers=8)

optimizer = AdamW(model.parameters(), lr=2e-5)

def train(model, loader, optimizer):
    model.train()
    loss_epoch = 0
    i = 0
    for X, y, mask in tqdm.tqdm(loader):
        i += 1
        optimizer.zero_grad()
        X = X.to(device)
        y = y.to(device)
        mask = mask.to(device)
        O = model(X, labels=y, attention_mask=mask)
        loss = O[0]
        loss_epoch += loss.item()
        loss.backward()
        optimizer.step()
        if i%500 == 0:
            print("Loss:", loss_epoch/i)
            torch.save(model.state_dict(), model_name)
    print ("Loss:", loss_epoch/len(loader))
    
    return loss_epoch/len(loader)

def test(model, loader):
    model.eval()
    outputs = []
    scores = []
    with torch.no_grad():
        for X, y, mask in tqdm.tqdm(loader):
            X = X.to(device)
            mask = mask.to(device)
            output = model(X, attention_mask=mask)
            scores.extend(output[0].detach().cpu().numpy()[:, 1])
            outputs.extend(output[0].detach().cpu().argmax(dim=1).numpy())
    
    return np.asarray(outputs), np.asarray(scores)

# Get top 5 evidences for each claim
def get_top_5(preds, scores, ids, predicted_evidence):
    
    evidence_map = {}
    top_5_map = {}
    
    for i in range(len(ids)):
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

for i in range(NUM_EPOCHS):
    x = train(model, train_loader, optimizer)
    torch.save(model.state_dict(), model_name)

# Train Set
preds, scores = test(model, train_loader)
top_5_map = get_top_5(preds, scores, ids_train, predicted_evidence_train)
format_output('train_sent_results.txt', top_5_map)

# Dev Set
preds, scores = test(model, dev_loader)
top_5_map = get_top_5(preds, scores, ids_dev, predicted_evidence_dev)
format_output('dev_sent_results.txt', top_5_map)

# Test Set
preds, scores = test(model, test_loader)
top_5_map = get_top_5(preds, scores, ids_test, predicted_evidence_test)
format_output('test_sent_results.txt', top_5_map)
