from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt  
import json
import numpy as np

def plot_confusion_matrix(y_true, y_pred, labels, plabels):
	mat = confusion_matrix(y_true, y_pred, labels=labels)
	mat = mat/np.sum(mat, axis=1)
	ax = plt.subplot()
	sns.heatmap(mat, annot=True, ax = ax)
	ax.set_xlabel('Predicted labels')
	ax.set_ylabel('True labels')
	ax.xaxis.set_ticklabels(plabels)
	ax.yaxis.set_ticklabels(plabels)
	plt.savefig("confusion_matrix.png")

y_true = []
y_pred = []

f = open("dev_results.txt")
for line in f:
	line = json.loads(line)
	y_true.append(line["label"])
	y_pred.append(line["predicted_label"])

plot_confusion_matrix(y_true, y_pred, ["NOT ENOUGH INFO", "SUPPORTS", 
	"REFUTES"], ["NEI", "SUPPORTS", "REFUTES"])
