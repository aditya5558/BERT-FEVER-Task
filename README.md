# BERT-FEVER-Task
Neural Networks for NLP Project

### Authors
Aditya Anantharaman (AndrewID: adityaan)  
Derik Clive Robert (AndrewID: dclive)  
Abhinav Khattar (AndrewID: akhattar)  

### General
The repo contains .py files required to run the code.  
To replicate results, run code in the following order:  
1. [doc_retrieval.py](https://github.com/aditya5558/BERT-FEVER-Task/blob/master/doc_retrieval.py): retrieves docs using the MediaWiki API  
2. [sentence_retrieval.py](https://github.com/aditya5558/BERT-FEVER-Task/blob/master/sentence_retrieval.py): retrieves top 5 sentences for every claim  
3. [claim_verification.py](https://github.com/aditya5558/BERT-FEVER-Task/blob/master/claim_verification.py): classifies the top 5 sentences for every claim  

[Bert.py](https://github.com/aditya5558/BERT-FEVER-Task/blob/master/Bert.py) contains our implementation of BertEncoder from scratch.

[tokenization.py](https://github.com/aditya5558/BERT-FEVER-Task/blob/master/tokenization.py) is taken from the official Bert repo and is used to tokenize data to feed to the models.

[scorer.py](https://github.com/aditya5558/BERT-FEVER-Task/blob/master/scorer.py) is the official scoring script provided by the authors of the FEVER Task.

[Utils folder](https://github.com/aditya5558/BERT-FEVER-Task/tree/master/utils) contains utilities to preprocess and normalize data, and generate helpful plots. You may require scripts from this folder to ensure that data is in the format our main code can parse.

[Auxiliary folder](https://github.com/aditya5558/BERT-FEVER-Task/tree/master/auxiliary) contains code that uses PyTorch transformer to achieve the same task, along with a few extra ipynb files. We have not used any of the scripts in this folder for getting our final results.

### Pipeline
![Pipeline](img/flowchart.png "Pipeline")

### Results
![Results](img/results.jpeg "Results")

### Report
All other description of the project along with the architecture is present in the report.pdf

### Reference
Implementation of technique proposed by:  
[BERT for Evidence Retrieval and Claim Verification](https://arxiv.org/pdf/1910.02655.pdf) by Soleimani et al.