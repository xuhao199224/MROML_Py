__author__ = 'hao'

from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


def loadDataSet(filename):
    dataMat = []
    fr = open(filename, 'UTF-8')
    for line in fr.readlines():
        # print line
        # if line.strip().find(',') != -1:
        #     curLine = line.strip().split(',')
        # else:
        curLine = line.strip().split('\t')
        # print curLine
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat


def loadBCDataSet(filename, skipLineNum):
    dataMat = []
    fr = open(filename, 'UTF-8')
    for line in fr.readlines():
        if skipLineNum > 0:
            skipLineNum -= 1
            continue
        curLine = line.strip().split(',')
        if curLine[3] >= '1':
            fltLine = map(str, curLine)
            dataMat.append(curLine)
    return dataMat


def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print centroids
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    while len(centList) < k:
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print "sseSplit, and notSplit: ", sseSplit, sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print 'the bestCentToSplit is: ', bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssment


# def clusterClubs(numClust=5):
#     datList = []
#     for line in open('places.txt').readlines():
#         lineArr = line.split('\t')
#         datList.append([float(lineArr[4]), float(lineArr[3])])
#     datMat = mat(datList)
#     myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
#     fig = plt.figure()
#     rect = [0.1, 0.1, 0.8, 0.8]
#     scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
#     axprops = dict(xticks=[], yticks=[])
#     ax0 = fig.add_axes(rect, label='ax0', **axprops)
#     imgP = plt.imread('Portland.png')
#     ax0.imshow(imgP)
#     ax1 = fig.add_axes(rect, label='ax1', frameon=False)
#     for i in range(numClust):
#         ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
#         markerStyle = scatterMarkers[i % len(scatterMarkers)]
#         ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
#                     s=90)
#     ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
#     plt.show()
#
#
# def distSLC(vecA, vecB):
#     a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
#     b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
#     return arccos(a + b) * 6371.0


def drawPlot(datMat):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = 100
    # for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    #     xs = randrange(n, 23, 32)
    #     ys = randrange(n, 0, 100)
    #     zs = randrange(n, zl, zh)
    #     ax.scatter(xs, ys, zs, c=c, marker=m)
    ax.scatter(datMat[:, 0].flatten().A[0], datMat[:, 1].flatten().A[0], datMat[:, 2].flatten().A[0], marker='+', s=300)
    plt.show()
    return



# dataMat = mat(loadDataSet('testSet.txt'))
# drawPlot(dataMat)