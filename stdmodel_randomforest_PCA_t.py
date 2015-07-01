from sklearn.datasets import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.metrics import roc_curve  
from sklearn.metrics import auc 
from sklearn.decomposition import PCA
import numpy
def run(fff1, fff2, fff3, fff4):
	X_train, y_train, X_test, y_test = load_svmlight_files((fff1,fff2))

	y_test = numpy.array(y_test)
	print y_test.shape

	proj = PCA(n_components = 15)
	X_train_new = proj.fit_transform(X_train.toarray())
	X_test_new = proj.transform(X_test.toarray())

	clsier = RandomForestClassifier(n_estimators = 20, min_samples_leaf=2, max_features = 20, verbose=1, n_jobs = -1)
	#clsier = DecisionTreeClassifier()
	clsier.fit(X_train_new, y_train)
	y_prob = numpy.array(clsier.predict_proba(X_test_new))
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
	plt.plot(fpr, tpr, linewidth=2, label = "randomforest_fea1_PCA")

	truth = open(fff3)
	pred = open(fff4)

	y = [float(line.split(' ',1)[0]) for line in truth]
	p = [float(line) for line in pred]

	fpr, tpr, thresholds = roc_curve(y, p, pos_label=1)  
	print auc(fpr, tpr)
	plt.plot(fpr, tpr, linewidth=2, label = "randomforest_fea2_PCA")
	plt.legend(fontsize=10, loc='best')
	plt.tight_layout()

	plt.savefig(rocfile)
#################
if __name__ == '__main__':
	run("./fea/train_fea1_1", "./fea/train_fea1_2", './fea/train_fea1_2', './localtest/pred_rf1_PCA')
	run("./fea/train_fea2_1", "./fea/train_fea2_2", './fea/train_fea2_2', './localtest/pred_rf2_PCA')
	eva("fea/train_fea1_2", "localtest/pred_rf1_PCA", "fea/train_fea2_2", "localtest/pred_rf2_PCA", "rocfigures/roc_rf1_PCA.jpg")