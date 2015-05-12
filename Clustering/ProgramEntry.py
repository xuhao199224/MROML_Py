__author__ = 'hao'


from numpy import *
import kMeans
import os


dataMat = mat(kMeans.loadDataSet('testSet2.txt'))
kMeans.drawPlot(dataMat)