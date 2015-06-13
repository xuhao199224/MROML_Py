__author__ = 'hao'

from numpy import *
import kMeans
import DBScan
from sklearn.preprocessing import StandardScaler


equipid = '11BC53130992'
# dataSet = mat(kMeans.loadDataSet('testSet.txt'))
dataSetOri = kMeans.loadBCDataSet('/Users/hao/Documents/github/BCData/datahao/' + equipid + '.csv', 1)

dataSet = []
for i in range(0, len(dataSetOri)):
    curLine = dataSetOri[i]
    print curLine
    if curLine[3] >= '1':
        fltLine = map(float, curLine[2:5])
        dataSet.append(fltLine)

dataSet = mat(dataSet)
dataSet_Scaled = StandardScaler().fit_transform(dataSet)

# for i in range(0, dataSet.shape[0]):
#     print dataSet[i, :]
#     print dataSet_Scaled[i, :]

DBScan.DBScan(0.3, 10, dataSetOri, dataSet_Scaled, equipid, True)