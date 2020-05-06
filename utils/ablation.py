import json
from collections import Counter

count = Counter()
f = open("dev-data.jsonl")
for line in f:
	line = json.loads(line)
	if line["label"] == "NOT ENOUGH INFO":
		count[line["id"]] = 1
	elif line["is_evidence"] == 1:
		count[line["id"]] = 1
	else:
		count[line["id"]] |= 0
f.close()

f = open("shared_task_dev.jsonl")
for line in f:
	line = json.loads(line)
	if line["id"] in count:
		continue
	if line["label"] == "NOT ENOUGH INFO":
		count[line["id"]] = 1
	else:
		count[line["id"]] |= 0
f.close()



print (sum(count.values())/len(count))