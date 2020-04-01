import numpy as np
import json, tqdm, sys
from matplotlib import pyplot as plt 
import seaborn as sns
from tokenization import FullTokenizer

vocab_file = "uncased_L-12_H-768_A-12/vocab.txt"

def plot_stats(lens):
	ax = sns.distplot(lens)
	ax.set_xlabel('Sentence Lengths')
	plt.savefig("stats.png")

def load_data(fname):
	f = open(fname, encoding='utf8')
	data = []
	for line in f:
		line = json.loads(line)
		sentence = ["[CLS]" + line['claim'] + "[SEP]", line['sentence'] + "[SEP]"]
		data.append(sentence)
	f.close()
	return data

def get_lens(data):
	tokenizer = FullTokenizer(vocab_file)
	lens = []
	for pos, text in tqdm.tqdm(enumerate(data)):
		tok0 = tokenizer.tokenize(text[0])
		tok1 = tokenizer.tokenize(text[1])
		tok = tok0 + tok1
		lens.append(len(tok))

	return np.array(lens)

method = "raw"
if len(sys.argv) > 1:
	method = sys.argv[1]

if method == "masks":
	print ("Generating plot from masks")
	masks = np.load("dev-masks-new.npy").reshape(-1,128)
	lens = np.sum(masks, axis=1)
elif method == "lens":
	print ("Generating plot from stored lens")
	lens = np.load("dev-lens.npy")
else:
	print ("Generating plot from raw data")
	print ("This may take some time...")
	data = load_data("dev-data.jsonl")
	lens = get_lens(data)
	np.save("dev-lens.npy", lens)

mean, std = np.mean(lens), np.std(lens)
print ("Mean, Std. Dev.", mean, std)


# lens = np.clip(lens, 0, 128)
plot_stats(lens)