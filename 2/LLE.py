# 作者:     wxf

# 开发时间: 2022/6/4 14:17
import csv
from numpy import *
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding

def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return
def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


X_mi = []
pro_X = []
ReadMyCsv(X_mi,'MiSimilarity.csv')
pro_X_mi = LocallyLinearEmbedding(n_components=64, n_neighbors = 50).fit_transform(X_mi)
StorFile(pro_X_mi, '64_X_mi.csv')

X_circ = []
pro_X_circ = []
ReadMyCsv(X_circ,'circSimilarity.csv')
pro_X_circ = LocallyLinearEmbedding(n_components=64, n_neighbors = 50).fit_transform(X_circ)
StorFile(pro_X_circ, '64_X_circ.csv')