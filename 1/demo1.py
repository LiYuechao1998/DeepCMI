# 作者:     wxf

# 开发时间: 2022/6/1 15:50
import numpy as np
import pandas as pd
import csv
import math
import random
import xlrd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from numpy import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
# 定义函数
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

MiSimilarity = []
jar = []
GIP = []
ReadMyCsv(jar, '962_2jar.csv')
ReadMyCsv(GIP, 'MIGaussian.csv')

counter = 0
while counter < len(jar):
    counter1 = 0
    Row = []
    while counter1 < len(jar):
        v = float(jar[counter][counter1])
        if v > 0:
            Row.append(v)
        if v == 0:
            Row.append(GIP[counter][counter1])
        counter1 = counter1 + 1
    MiSimilarity.append(Row)
    counter = counter + 1
print('len(MiSimilarity)', len(MiSimilarity))
print('len(MiSimilarity[0])',len(MiSimilarity[0]))
storFile(MiSimilarity, 'MiSimilarity.csv')

#circ

CircSimilarity = []
jarC = []
GIPC = []
ReadMyCsv(jarC, '2346_5jar.csv')
ReadMyCsv(GIPC, 'CIRCGaussian.csv')

counter = 0
while counter < len(jarC):
    counter1 = 0
    Row = []
    while counter1 < len(jarC):
        v = float(jarC[counter][counter1])
        if v > 0:
            Row.append(v)
        if v == 0:
            Row.append(GIPC[counter][counter1])
        counter1 = counter1 + 1
    CircSimilarity.append(Row)
    counter = counter + 1
print('len(CircSimilarity)', len(CircSimilarity))
print('len(CircSimilarity[0])',len(CircSimilarity[0]))
storFile(CircSimilarity, 'circSimilarity.csv')