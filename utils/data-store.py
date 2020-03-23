import json, tqdm

def extract_data():
	N = 109
	data = {}

	for i in tqdm.tqdm(range(1, N+1)):
		f = open("wiki-pages/wiki-"+str(i).zfill(3)+".jsonl", encoding='utf8')
		for line in f:
			line = json.loads(line)
			data[line['id']] = line['lines']
		f.close()

	print ("Number of dump entries",len(data))

	return data

def extract_info(fname):
	info = {}

	f = open(fname, encoding='utf8')
	for line in f:
		line = json.loads(line)
		ev_set = set()
		if 'evidence' in line:
			for ev in line['evidence']:
				for val in ev:
					ev_set.add((val[2], str(val[3])))

		if 'label' in line:
			info[line['id']] = [line['label'], ev_set]
		else:
			info[line['id']] = ['UNK', ev_set]
	f.close()

	return info

def generate_file(fname1, fname2, wname):
	data = extract_data()
	info = extract_info(fname1)
	extracted = []

	f = open(fname2, encoding='utf8')
	for line in f:
		line = json.loads(line)
		extracted.append(line)
	f.close()

	f = open(wname, "w", encoding='utf8')
	output = []
	for val in tqdm.tqdm(extracted):
		res = {}
		res['id'] = val['claim_id']
		res['claim'] = val['claim']
		res['label'] = info[res['id']][0]
		insert = 0
		for doc in val['docs']:
			doc = doc.replace("e\u0301", "\u00e9")
			if doc not in data:
				continue
			insert = 1
			res['doc'] = doc
			lines = data[doc].split("\n")
			for line in lines[:-1]:
				spline = line.split("\t")
				try:
					res['sid'] = int(spline[0])
				except:
					print ("Uneven split at:", res['id'],doc)
					continue
				res['sentence'] = ("\t").join(spline[1:])
				res['is_evidence'] = 0
				if (doc, str(spline[0])) in info[res['id']][1]:
					res['is_evidence'] = 1
				json.dump(res, f)
				f.write("\n")
		# if not insert and val['docs'] != []:
		# 	print ("No doc found for:", res['id'])
	f.close()


if __name__ == "__main__":
	generate_file("shared_task_test.jsonl", "test_out.txt", "test-data.jsonl")