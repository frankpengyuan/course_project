from sklearn.datasets import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.metrics import roc_curve  
from sklearn.metrics import auc 
from sklearn.svm import *
import numpy
def complex(fff1, fff2):
	file1 = open(fff1)
	file2 = open(fff2)
	cnt = 0
	y = []
	for line in file1:
		cnt = cnt + 1
	file1.close()
	file1 = open(fff1)
	p1 = [float(line) for line in file1]
	p2 = [float(line) for line in file2]
	for x in range(0,cnt):
		if p1[x] > 0.5 and p2[x] > 0.5:
			y.append(1.0-(1.0-p1[x])*(1.0-p2[x]))
		elif p1[x] > 0.5 and p2[x] > 0.5:
			y.append(p1[x]*p2[x])
		else:
			y.append(p1[x]+p2[x]/2.0)
	for i in y:
		if i > 0.85:
			i = 1.0
		elif i < 0.15:
			i = 0.1
	return y


import matplotlib.pyplot as plt

def eva_complex(fff1, y1, fff3, y2, rocfile):
	truth = open(fff1)

	y = [float(line.split(' ',1)[0]) for line in truth]
	p = y1

	fpr, tpr, thresholds = roc_curve(y, p, pos_label=1)  
	print auc(fpr, tpr)

	plt.figure(figsize=(4, 4), dpi=80)
	x = [0.0, 1.0]
	plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')

	plt.xlim(0.0, 1.0)
	plt.ylim(0.0, 1.0)
	plt.xlabel("FPR", fontsize=14)
	plt.ylabel("TPR", fontsize=14)
	plt.title("ROC Curve", fontsize=14)
	plt.plot(fpr, tpr, linewidth=2, label = "complex_allfea")
	'''
	truth = open(fff3)

	y = [float(line.split(' ',1)[0]) for line in truth]
	p = y2

	fpr, tpr, thresholds = roc_curve(y, p, pos_label=1)  
	print auc(fpr, tpr)
	plt.plot(fpr, tpr, linewidth=2, label = "complex_fea2")
	'''
	plt.legend(fontsize=10, loc='best')
	plt.tight_layout()

	plt.savefig(rocfile)
#################
if __name__ == '__main__':
	y1 = complex("./localtest/pred_rf1", "./localtest/pred_rf1")
	y2 = complex("./localtest/pred_rf2", "./localtest/pred_csvm2")
	eva_complex("fea/train_fea1_2", y1, "fea/train_fea2_2", y2, "rocfigures/roc_complex1.jpg")