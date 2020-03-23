import json, tqdm


N = 109
resl = []

for i in tqdm.tqdm(range(1, N+1)):
	f = open("wiki-pages/wiki-"+str(i).zfill(3)+".jsonl", encoding='utf8')
	for line in f:
		line = json.loads(line)
		resl.append(line['id'])
	f.close()

f = open("IDs.txt", "w", encoding='utf8')
f.write(str(resl))
f.close()
