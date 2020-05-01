from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
import torch, tqdm, json
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset 
from scorer import fever_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 2

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
model.to(device)

model_name = "ClaimVerification.pt"

def process_data(fname, train=True):
    X = []
    y = []

    label_dict = {}
    label_dict['UNK'] = -1
    label_dict['NOT ENOUGH INFO'] = 0
    label_dict['SUPPORTS'] = 1
    label_dict['REFUTES'] = 2
    claim_ids = []
    f = open(fname, encoding='utf8')
    f.readline()
    maxi = 0
    preds_dict = {}
    for line in f:
        line = json.loads(line)
        claim_ids.append(line['id'])
        preds_dict[line["id"]] = [line["label"], line["predicted_evidence"]]

        input_ids = (tokenizer.encode(line['claim']+" [SEP] "+line["sentence"], add_special_tokens=True))[:256]
        X.append(input_ids)
        if len(input_ids) > maxi:
            maxi = len(input_ids)
        y.append(label_dict[line['label']])
    f.close()
    mask = []
    for val in X:
        mask.append([1]*len(val) + [0]*(maxi-len(val)))
        val += [0]*(maxi-len(val)) #np.pad(val, (0, maxi-len(val)), 'constant')
        
    return torch.LongTensor(X), torch.LongTensor(y), torch.LongTensor(mask), claim_ids, preds_dict


X_train, y_train, mask_train, _, preds_dict_train = process_data("train_sent_multi.txt")
X_dev, y_dev, mask_dev, ids_dev, preds_dict_dev = process_data("dev_sent_multi.txt")
X_test, y_test, mask_test, ids_test, preds_dict_test = process_data("test_sent_multi.txt")

train_dataset = TensorDataset(X_train, y_train, mask_train)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32, num_workers=8)

dev_dataset = TensorDataset(X_dev, y_dev, mask_dev)
dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=32, num_workers=8)

test_dataset = TensorDataset(X_test, y_test, mask_test)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32, num_workers=8)

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
            outputs.extend(output[0].detach().cpu().argmax(dim=1).numpy())
    
    return np.asarray(outputs)

# Merge predictions for each claim
def merge_preds(preds, ids, preds_dict):
    label_dict = {}
    label_dict[0] = 'NOT ENOUGH INFO'
    label_dict[1] = 'SUPPORTS'
    label_dict[2] = 'REFUTES'
    for i in range(len(ids)):
        preds_dict[ids[i]] = [label_dict[preds[i]], preds_dict[ids[i]][1]]
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

for i in range(NUM_EPOCHS):
    x = train(model, train_loader, optimizer)
    torch.save(model.state_dict(), model_name)

# print('Loading model')
# model.load_state_dict(torch.load(model_name))
# model.to(device)

# Dev Set
preds = test(model, dev_loader)
preds_dict = merge_preds(preds, ids_dev, preds_dict_dev)
format_output('dev.jsonl', 'dev_results.txt', preds_dict)


# Test Set
preds = test(model, test_loader)
preds_dict = merge_preds(preds, ids_test, preds_dict_test)
format_output('test.jsonl', 'test_results.txt', preds_dict, dev=False)

