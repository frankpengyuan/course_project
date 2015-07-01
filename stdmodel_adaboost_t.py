from sklearn.datasets import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.metrics import roc_curve  
from sklearn.metrics import auc 
import numpy
def run(fff1, fff2, fff3, fff4):
	X_train, y_train, X_test, y_test = load_svmlight_files((fff1,fff2))

	y_test = numpy.array(y_test)
	print y_test.shape

	clsier = AdaBoostClassifier(n_estimators = 60)
	#clsier = DecisionTreeClassifier()
	clsier.fit(X_train, y_train)
	y_prob = numpy.array(clsier.predict_proba(X_test))
	print y_prob
	y_pred = y_prob[:,1]

	sample_submission_file = open(fff3)
	submission_file = open(fff4,'w')
	cnt = 0
	for line in sample_submission_file:
		new_line = str(y_pred[cnt]) + '\n'
		submission_file.write(new_line)
		cnt += 1
	print cnt
	sample_submission_file.close()
	submission_file.close()


import matplotlib.pyplot as plt

def eva(fff1, fff2, fff3, fff4, rocfile):
	truth = open(fff1)
	pred = open(fff2)

	y = [float(line.split(' ',1)[0]) for line in truth]
	p = [float(line) for line in pred]

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
	plt.plot(fpr, tpr, linewidth=2, label = "adaboost_fea1")

	truth = open(fff3)
	pred = open(fff4)

	y = [float(line.split(' ',1)[0]) for line in truth]
	p = [float(line) for line in pred]

	fpr, tpr, thresholds = roc_curve(y, p, pos_label=1)  
	print auc(fpr, tpr)
	plt.plot(fpr, tpr, linewidth=2, label = "adaboost_fea2")
	plt.legend(fontsize=10, loc='best')
	plt.tight_layout()

	plt.savefig(rocfile)
#################
if __name__ == '__main__':
	run("./fea/train_fea1_1", "./fea/train_fea1_2", './fea/train_fea1_2', './localtest/pred_ada1')
	run("./fea/train_fea2_1", "./fea/train_fea2_2", './fea/train_fea2_2', './localtest/pred_ada2')
	eva("fea/train_fea1_2", "localtest/pred_ada1", "fea/train_fea2_2", "localtest/pred_ada2", "rocfigures/roc_ada1.jpg")