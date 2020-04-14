from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
import json
from multiprocessing.dummy import Pool as ThreadPool
import tqdm
import unicodedata
import wikipedia

####### Doc Retrieval Accuracy: 83.46% 

archive = load_archive("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
predictor = Predictor.from_archive(archive, 'constituency-parser')

# Get doc ids in wikipedia dump
with open('IDs.txt', 'r') as f:
		doc_ids = f.read()

# Extract Noun Phrases from Constituency Parse Tree
def extract_noun_phrases(root):
	np_list = []
	stack = [root]
	while len(stack):
		top = stack.pop()
		if top['nodeType'] == 'NP':
			np_list.append(top['word'])

		if 'children' in top.keys():
			for child in top['children']:
				stack.append(child)
	return np_list

# Extract entities before main verb
def extract_other_entities(root, claim):
	other_entities = []
	words = []
	claim_tokens = word_tokenize(claim) 
	for word in claim_tokens:
		nodeType = find_word(root, word)
		if nodeType == 'VB' or nodeType == 'ADVP' or nodeType == 'VP' or nodeType == 'VBD':
			if len(words):
				other_entities.append(' '.join(words))
			words.append(word)
		else:
			words.append(word)
	return other_entities

# Find Nodetype of a word in Constituency Tree
def find_word(root, word):
	stack = [root]
	while len(stack):
		top = stack.pop()
		if top['word'] == word:
			return top['nodeType']
		if 'children' in top.keys():
			for child in top['children']:
				stack.append(child)
	return None

# Perform search on entities using mediawiki api
def wiki_search(claim, entities):
	k = 7
	documents = []
	for entity in entities:
		try:
			if entity is not None:
				results = wikipedia.search(entity)
				documents.extend(results[:k])
		except:
			print(entity)
			print(claim)

	return list(set(documents))

# Function to get doc/claim in proper dataset format
def repl(doc):
	doc = doc.replace(' ', '_')
	doc = doc.replace('(', '-LRB-')
	doc = doc.replace(')', '-RRB-')
	doc = doc.replace(':', '-COLON-')
	return doc

# Filter out documents (Step 3: Candidate Filtering)
def filter(documents, claim):
	final_documents = []
	stemmer = PorterStemmer() 
	claim_tokens = word_tokenize(claim) 
	claim_tokens = [stemmer.stem(token.lower()) for token in claim_tokens]
	claim_tokens = list(set(claim_tokens))

	for doc in documents:
		left = doc.find('(')
		right = doc.find(')')
		if (left != -1 and right != -1):
			new_doc = doc[:left] + doc[right+1:]
		else:
			new_doc = doc
		doc_tokens = word_tokenize(new_doc) 
		doc_tokens = [stemmer.stem(token.lower()) for token in doc_tokens]
		doc_tokens = list(set(doc_tokens))

		include_flag = True
		for token in doc_tokens:
			if token not in claim_tokens:
				include_flag = False
				break	

		if include_flag:
			doc = repl(doc)
			doc = unicodedata.normalize('NFD', doc)
			final_documents.append(doc)
	return final_documents

# Retrieve documents using 3 steps
def doc_retriever(claim):
	# Step 1: Mention Extraction
	parse = predictor.predict(claim)
	root = parse['hierplane_tree']['root']

	noun_phrases = extract_noun_phrases(root)
	other_entities = extract_other_entities(root, claim)

	entities = []
	entities.extend(noun_phrases)
	entities.extend(other_entities)
	entities.append(claim)
	entities = list(set(entities))

	# Step 2: Candidate Article Search
	documents = wiki_search(claim, entities)
	dump_docs = extract_from_dump(noun_phrases)
	documents.extend(dump_docs)

	# Step 3: Candidate Filtering
	final_documents = filter(documents, claim)
	return final_documents

# Compute document retrieval accuracy on verifiable claims
def doc_retrieval_acc(file_path):
	evidence_sets = []
	claims = []
	with open(file_path, 'r') as f:
		for line in f.readlines():
			if json.loads(line.strip())['verifiable'] == 'VERIFIABLE':
				evidence_sets.append(json.loads(line.strip())['evidence'])
				claims.append(json.loads(line.strip())['claim'])

	reference_docs = []
	for evidence_set in evidence_sets:
		ref = []
		for evidence in evidence_set:
			for line in evidence:
				if line[2]:
					ref.append(line[2])
		reference_docs.append(list(set(ref)))

	processed_claims = len(claims)
	print("Total: {}".format(processed_claims))

	# Use thread pool to parallelize
	pool = ThreadPool(16)
	predicted_docs = list(tqdm.tqdm(pool.imap(doc_retriever, claims), total=processed_claims))
	pool.close()
	pool.join()

	correct = 0
	for j in range(processed_claims):
		add = True
		for doc in reference_docs[j]:
			if doc not in predicted_docs[j]:
				add = False
				break
		if add:
			correct += 1

	print("Correct: {}".format(correct))
	print("Document Retrieval Accuracy: {}".format(correct/processed_claims))

# Write document ids (names) to file in json format
def write_to_file(in_path, out_path):
	claims = []
	with open(in_path, 'r') as f:
		for line in f.readlines():
			claims.append(json.loads(line.strip())['claim'])

	processed_claims = len(claims)
	print("Total: {}".format(processed_claims))

	# Use thread pool to parallelize
	pool = ThreadPool(16)
	predicted_docs = list(tqdm.tqdm(pool.imap(doc_retriever, claims), total=processed_claims))
	pool.close()
	pool.join()

	for i in range(processed_claims):
		file_dict = {}
		file_dict['claim'] = claims[i]
		file_dict['docs'] = predicted_docs[i]

		with open(out_path, 'a+') as fout:
			json.dump(file_dict, fout)
			fout.write('\n')

# Extract documents from wikipedia 2017 dump provided in dataset
def extract_from_dump(entities):
	entities = [repl(entity) for entity in entities]
	docs = []
	for entity in entities:
		if entity in doc_ids:
			docs.append(entity)
	return docs

if __name__ == '__main__':
	
	# Sample Claims for testing

	claim1 = "Colin Kaepernick became a starting quarterback during the 49ers 63rd season in the National Football League."
	claim2 = "Down With Love is a 2003 comedy film."	
	claim3 = "Wish Upon starred a person."

	print(doc_retriever(claim1))

	in_path = 'data/fever-data/dev.jsonl'
	out_path = 'dev_out.txt'

	doc_retrieval_acc(in_path)
	write_to_file(in_path, out_path)







