from sklearn.datasets import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.metrics import roc_curve  
from sklearn.metrics import auc 
import numpy

X_train, y_train, X_test, y_test = load_svmlight_files(("./fea/train_fea1","./fea/test"))

y_test = numpy.array(y_test)
print y_test.shape

clsier = ExtraTreesClassifier(n_estimators = 100, min_samples_leaf=5, verbose=1, n_jobs = -1)
#clsier = DecisionTreeClassifier()
clsier.fit(X_train, y_train)
y_prob = numpy.array(clsier.predict_proba(X_test))
print y_prob
y_pred = y_prob[:,1]

sample_submission_file = open('raw/sampleSubmission.csv')
submission_file = open('sub/submission_extratrees.csv','w')
cnt = 0
for line in sample_submission_file:
	enroll_id = line.split(',')[0]
	new_line = enroll_id + ',' + str(y_pred[cnt]) + '\n'
	submission_file.write(new_line)
	cnt += 1
print cnt
sample_submission_file.close()
submission_file.close()
