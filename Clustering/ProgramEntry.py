__author__ = 'hao'

from numpy import *
import kMeans


dataMat = mat(kMeans.loadDataSet('testSet2.txt'))
kMeans.drawPlot(dataMat)