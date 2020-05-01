
import jsonlines
import json
from collections import Counter

# claim_dict = {}

# with jsonlines.open('test.jsonl') as reader:
# 	for obj in reader:
# 		if obj["label"] != "NOT ENOUGH INFO":
# 			for evidence_set in obj["evidence"]:
# 				for evidence in evidence_set:
# 					if obj["id"] in claim_dict :
# 						claim_dict[obj["id"]].append([evidence[2],evidence[3]])
# 					else:
# 						claim_dict[obj["id"]] = [[evidence[2],evidence[3]]]

# print(len(claim_dict))

train_sent_dict = {}
ids = {}
outputs = []
prev_id = -1
with jsonlines.open('test_sent_results.txt') as reader:
	for obj in reader:
		if obj["id"] not in ids:
			ids[obj["id"]] = 1
		if prev_id == -1:
			prev_id = obj["id"]

		if obj["id"] != prev_id:
			outputs.append(train_sent_dict)
			train_sent_dict = {}
			prev_id = obj["id"]

		train_sent_dict["id"] = obj["id"]
		train_sent_dict["claim"] = obj["claim"]

		if "label" in train_sent_dict:
			if train_sent_dict["label"] == "NOT ENOUGH INFO":
				train_sent_dict["label"] = obj["label"]
			elif train_sent_dict["label"] == "REFUTES" and obj["label"] == "SUPPORTS":
				train_sent_dict["label"] = obj["label"]
		else:
			train_sent_dict["label"] = obj["label"]

		if "predicted_evidence" in train_sent_dict:
			train_sent_dict["predicted_evidence"].append([obj['doc'], obj['sid']])
		else:
			train_sent_dict["predicted_evidence"] = [[obj['doc'], obj['sid']]]

		if "sentence" in train_sent_dict:
			train_sent_dict["sentence"] += " [SEP] "+obj["doc"]+obj["sentence"]
		else:
			train_sent_dict["sentence"] = obj["doc"]+obj["sentence"]

outputs.append(train_sent_dict)

with open("test_sent_new.txt", 'w', encoding='utf8') as f:
	for line in outputs:
		json.dump(line, f)
		f.write("\n")
print(len(ids))
