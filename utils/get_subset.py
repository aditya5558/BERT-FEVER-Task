import json, tqdm, random

f = open("train-data.jsonl", encoding='utf8')
f2 = open("train-data-sent-4.jsonl", "w", encoding='utf8')

flag = 0
for line in f:
	line = json.loads(line)
	p = random.random()
	if line["label"] == "NOT ENOUGH INFO": #and p<0.97:
		continue
	# if line["is_evidence"] == 0 and flag == 0:
	# 	continue
	if line["label"] == "SUPPORTS":
		continue
	if line["is_evidence"] == 0:
		if p < 0.98:
			continue
		flag = 0
		line["label"] = "NOT ENOUGH INFO"
	else:
		flag = 1
	json.dump(line, f2)
	f2.write("\n")

f2.close()
f.close()

