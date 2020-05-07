
import jsonlines
import json
from collections import Counter

claim_dict = {}
label_dict = {}
with jsonlines.open('train.jsonl') as reader:
	for obj in reader:
		label_dict[obj["id"]] = obj["label"]
		if obj["label"] != "NOT ENOUGH INFO":
			for evidence_set in obj["evidence"]:
				for evidence in evidence_set:
					if obj["id"] in claim_dict :
						claim_dict[obj["id"]].append([evidence[2],evidence[3]])
					else:
						claim_dict[obj["id"]] = [[evidence[2],evidence[3]]]

outputs = []
with jsonlines.open('train_sent_results.txt') as reader:
	for obj in reader:
		if obj["label"] != "NOT ENOUGH INFO":
			line = [obj["doc"], obj["sid"]]
			if line in claim_dict[obj["id"]]:
				obj["label"] = label_dict[obj["id"]]
		outputs.append(obj)

with open("train_sent_n.txt", 'w', encoding='utf8') as f:
	for line in outputs:
		json.dump(line, f)
		f.write("\n")
