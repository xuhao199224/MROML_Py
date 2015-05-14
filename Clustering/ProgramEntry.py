__author__ = 'hao'

from numpy import *
import kMeans
import DBScan
from sklearn.preprocessing import StandardScaler


dataSet = mat(kMeans.loadDataSet('testSet.txt'))
dataSet_Scaled = StandardScaler().fit_transform(dataSet)
DBScan.DBScan(0.3, 10, dataSet_Scaled)