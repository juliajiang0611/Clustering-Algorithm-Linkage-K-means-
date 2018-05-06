
# coding:UTF-8
import numpy as np
import pandas as pd
from numpy  import *
from random import random
from matplotlib import pyplot as plt



def loadDataSet(filename,delecol,label):
    readFile =pd.read_csv(filename,sep=',',header=None,dtype=str,na_filter=False)
    labelCol =readFile.iloc[:,label].copy()
    attrCol = readFile.drop(readFile.columns[delecol], axis=1).copy()
    data = np.array(attrCol).astype('float')
    return data,np.array(labelCol).astype(np.str)



# calculate Euclidean distance between vectors

def distEclud(v1, v2):
    return sqrt(sum(power(v1 - v2, 2)))

def nearestCenter(dataset, clusterCenteroids):
    minDis = inf
    # The number of clustering centers that are already initialized.
    m = np.shape(clusterCenteroids)[0]
    for i in xrange(m):
        # Calculate the distance between point and each cluster center.
        distance =sqrt(sum(power(dataset - clusterCenteroids[i, ], 2)))
        # choose the mininum distance
        if minDis > distance:
            minDis = distance
    return minDis


def getCentroids(dataSet, k):
    numRow = np.shape(dataSet)[0]
    numAttr = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k , numAttr)))
    # random choose a sample as first center
    index = np.random.randint(0, numRow)
    centroids[0, ] = np.copy(dataSet[index, ])
    # initalized a sequence for distance
    distance = [0.0 for i in range(numRow)]

    for i in range(1, k):

        sum = 0
        for j in range(numRow):
            # find the nearest center for each sample
            distance[j] = nearestCenter(dataSet[j, ], centroids[0:i, ])
            # add all min distance
            sum += distance[j]
        # get random value form sum

        sum *= random()
        # get the farthest sample point as the cluster center point.
        for j, d in enumerate(distance):
            sum -= d
            if sum > 0:
                continue
            centroids[i] = np.copy(dataSet[j, ])
            break
    return centroids


def kmeanspp(dataSet,k):
    # number of row
    numberRow = shape(dataSet)[0]
    # data assigned to cluster
    clusterAssment = np.mat(np.zeros((numberRow, 2)))
    cluster = [None] * dataSet.shape[0]
    # print cluster
    #initialized the centroids
    centroids = getCentroids(dataSet,k)
    clusterChanged = True
    while clusterChanged:

        clusterChanged = False
        for i in range(numberRow):
            # initial min to infinite
            minDist = inf
            minIndex = -1
            # cluster
            for j in range(k):
                # 1,3,4,5,8,9,10
                distance = sqrt(sum(power(centroids[j, :] - dataSet[i, :], 2)))
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True

            clusterAssment[i, :] = minIndex, minDist ** 2
            cluster[i] = minIndex

        # recalculate the new centroids based on cluster info
        for cent in range(k):
            pointInCluster = dataSet[nonzero(clusterAssment[:, 0].A == (cent))[0]]
            if len(pointInCluster):
                 centroids[(cent), :] = mean(pointInCluster, axis=0)

    return clusterAssment, cluster


def hammingDistance(label, target):
    distance = 0.0
    numRow = label.shape[0]
    for i in range(numRow):
        for j in range(numRow):
            if i != j :
                if label[i] == label[j] and target[i] != target[j]:
                    distance+=1
                if label[i] != label[j] and target[i] == target[j]:
                    distance+=1
    return distance/(numRow*(numRow-1))

def show(dataSet,clusterAssment):

    numSamples, dim = dataSet.shape
    mark =  ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr', 'xr', 'sb', 'sg', 'sk', '2r', '<b', '<g', '+b', '+g', 'pb']
    for i in xrange(numSamples):
       markIndex = int(clusterAssment[i, 0])
       plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
    plt.show()

if __name__ == "__main__":

    #------------------------------Dataset 1----------------------------------#
    deleteCols = [0]
    # [0,2,3,4,5,6,8,9,10,11,12,13]
    data, label = loadDataSet('wine.csv', deleteCols, 0)
    clusterAssment, clusters = kmeanspp(data, 3)
    #print clusters
    hammingDis = hammingDistance(label, clusters)

    # print clusters
    print "The hammingDistance of the first dataset : ", hammingDis
    #show(data, clusterAssment)

    #------------------------------Dataset 2-----------------------------------#
    #deleteCol = [0,1,2,4,5,6,8,9]
    deleteCol =[9]
    #[1,2,4,5,6,7,8,9]
    hammingDis= 0.0
    data,label = loadDataSet('glass.csv', deleteCol, 9)

    clusterAssement, clusters = kmeanspp(data,7)
    hammingDis += hammingDistance(label,clusters)
    # print clusters
    print "The hammingDistance of the first dataset : ", hammingDis
    #show(data,clusterAssement)







