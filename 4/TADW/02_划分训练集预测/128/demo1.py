import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from sklearn import linear_model

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import csv
import math
import random
import xlrd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from numpy import *
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import roc_curve, auc
# from RotationForest import RotationForest
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier(错的。。。。。。。。。。。。。。。。。。)
from xgboost.sklearn import XGBClassifier
from sklearn import metrics


def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return

def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

#*****************************标签******************************************
cv = StratifiedKFold(n_splits=5)
data_label = []
data1 = []
data1 = ones((1,9905), dtype=int)
# 2532 1933
data2 = zeros((1, 9905))
data_label.extend(data1[0])
data_label.extend(data2[0])
SampleLabel = data_label
# print(SampleLabel)
# storFile(SampleLabel, 'SampleLabel.csv')

#***************************划分训练集************************************
SampleFeature = []
ReadMyCsv(SampleFeature, 'SampleFeature.csv')
# SampleFeature = np.array(SampleFeature)
# print(np.shape(SampleFeature))
# x_train, x_test, y_train, y_test = train_test_split(SampleFeature, SampleLabel, test_size=0.2)
# print(np.shape(x_train))
# y_train = np.array(y_train)
# y_test = np.array(y_test)
print('Start training the model.')
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 10000)
i = 0
num=0
cv = StratifiedKFold(n_splits=5)
SampleFeature = np.array(SampleFeature)  #打乱顺序
SampleLabel = np.array(SampleLabel)
permutation = np.random.permutation(SampleLabel.shape[0])
SampleFeature = SampleFeature[permutation, :]
SampleLabel = SampleLabel[permutation]

Y_test = []
Y_pre = []
for train, test in cv.split(SampleFeature, SampleLabel):
    # print(np.shape(SampleFeature[train]))
    # print(SampleFeature[train][:,1:])
    storFile(SampleFeature[test][:,0:1],'第'+str(i)+'折测试集合序号')
    # SampleFeature[train] = SampleFeature[train][:,1:]
    # SampleFeature[test] = SampleFeature[test][:,1:]

    # SampleFeature[train] = np.array(SampleFeature[train])
    # SampleFeature[train] = SampleFeature[train][,1:]
    # print(np.shape(SampleFeature[train]))



    # logistic = linear_model.LogisticRegression()
    # rbm = BernoulliRBM(random_state=0, verbose=True)
    # NN_clf = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    # rbm.learning_rate = 0.1
    # rbm.n_iter = 20
    # rbm.n_components = 100
    # #正则化强度参数
    # logistic.C = 1000
    # NN_clf.fit(SampleFeature[train],np.array(SampleLabel[train]))
    # # predicted1 = np.around(NN_clf.predict_proba(SampleFeature[test]), 0).astype(int)
    #
    # predicted1 =  NN_clf.predict_proba(SampleFeature[test])
    y_test1 = []
    y_pre1 = []

    # model = KNeighborsClassifier(n_neighbors=3)
    # model = GradientBoostingClassifier(n_estimators=10)
    # model = GradientBoostingClassifier(n_estimators=10)
    model = XGBClassifier(max_depth=6,n_estimators=2)
    # # model = RotationForest(n_estimators=139)
    # # model = RotationForest(n_classifiers=139)
    predicted = model.fit(SampleFeature[train][:,1:], SampleLabel[train]).predict_proba(SampleFeature[test][:,1:])
    # predicted1 = model.fit(SampleFeature[train], SampleLabel[train]).predict_proba(SampleFeature[test])



    fpr, tpr, thresholds = roc_curve(SampleLabel[test], predicted[:, 1])


    predicted1 = model.predict(SampleFeature[test][:,1:])
    num = num + 1
    print("==================", num, "fold", "==================")
    print('Test accuracy: ', accuracy_score(SampleLabel[test], predicted1))
    print(classification_report(SampleLabel[test], predicted1, digits=4))
    print(confusion_matrix(SampleLabel[test], predicted1))

    con = []
    con.append(accuracy_score(SampleLabel[test], predicted1))
    np.save('第' + str(i)+'折acc', con)

    con1 = []
    con1.append(classification_report(SampleLabel[test], predicted1, digits=4))
    np.save('第' + str(i)+'折各项参数', con1)

    con2 = []
    con2.append(confusion_matrix(SampleLabel[test], predicted1))
    np.save('第' + str(i)+'折混淆矩阵', con2)

    Y_test.extend(SampleLabel[test])
    Y_pre.extend(predicted1)

    y_test1.extend(SampleLabel[test])
    y_pre1.extend(predicted[:, 1])
    np.save('Y_test'+str(i),y_test1)
    np.save('Y_pre' + str(i), y_pre1)

    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.5,
             label='ROC fold %d (AUC = %0.4f)' % (i, roc_auc))

    i += 1

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
# mean_auc = metrics.auc(mean_fpr, mean_tpr)
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc), lw=1.5, alpha=1)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate',fontsize=13)
plt.ylabel('True Positive Rate',fontsize=13)
plt.title('ROC',fontsize=13)
plt.legend(loc="lower right")
plt.savefig('ROC-5fold.tif')
plt.show()

sns.set()
f, ax = plt.subplots()
y_true = np.array(Y_test)
y_pred = np.array(Y_pre)
C2 = confusion_matrix(y_true, y_pred)
# 打印 C2
print(C2)
sns.heatmap(C2, annot = True, ax = ax)  # 画热力图

ax.set_xlabel('predict')  # x 轴
ax.set_ylabel('true')  # y 轴
plt.savefig('confusion_matrix-5fold1.tif')
plt.show()

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# from xgboost import XGBClassifier

# models = []
# models.append(("LR", LogisticRegression())) #逻辑回归
# models.append(("NB", GaussianNB())) # 高斯朴素贝叶斯
# models.append(("KNN", KNeighborsClassifier())) #K近邻分类
# models.append(("DT", DecisionTreeClassifier())) #决策树分类
# models.append(("SVM", SVC())) # 支持向量机分类
# models.append(("xgboost", XGBClassifier())) #支持向量机分类

# for train, test in cv.split(SampleFeature, SampleLabel):
#     model = XGBClassifier(n_estimators=10)
#     predicted = model.fit(np.array(SampleFeature[train]), np.array(SampleLabel[train])).predict_proba(np.array(SampleFeature[test]))
#
#
#     fpr, tpr, thresholds = roc_curve(SampleLabel[test], predicted[:, 1])
#
#     predicted1 = model.predict(SampleFeature[test])
#     num = num + 1
#     print("==================", num, "fold", "==================")
#     print('Test accuracy: ', accuracy_score(SampleLabel[test], predicted1))
#     print(classification_report(SampleLabel[test], predicted1, digits=4))
#     print(confusion_matrix(SampleLabel[test], predicted1))
#
#     tprs.append(interp(mean_fpr, fpr, tpr))
#     tprs[-1][0] = 0.0
#     roc_auc = auc(fpr, tpr)
#     aucs.append(roc_auc)
#     plt.plot(fpr, tpr, lw=1, alpha=0.3,
#              label='ROC fold %d (AUC = %0.4f)' % (i, roc_auc))
#
#     i += 1
#
# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# std_auc = np.std(aucs)
# plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc), lw=1.5, alpha=.8)
#
# std_tpr = np.std(tprs, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
#
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()



