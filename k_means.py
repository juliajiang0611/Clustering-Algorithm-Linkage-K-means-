# coding:UTF-8
import numpy as np
from numpy import *
import pandas as pd
import random
from matplotlib import pyplot as plt

def loadDataset(filename,delecol,label):
    readFile =pd.read_csv(filename,sep=',',header=None,dtype=str,na_filter=False)
    labelCol =readFile.iloc[:,label].copy()
    attrCol = readFile.drop(readFile.columns[delecol], axis=1).copy()
    data = np.array(attrCol).astype('float')
    return data, np.array(labelCol).astype(np.str)


def randomCenter(data,k):
    # n is the number of attributes
    n = np.shape(data)[1]
    #Initialize k centroids
    centroids = np.mat(np.zeros((k, n)))
    # Initializes the coordinates of each dimension of the cluster center.
    for j in range(n):
        min = np.min(data[:, j])
        difference = np.max(data[:, j]) - min
        # Random initialization between maximum and minimum values.
        centroids[:, j] = min * np.mat(np.ones((k, 1))) + np.random.rand(k, 1) * difference
    return centroids



def kmeans(dataSet,k):
    # number of row
    numberRow = shape(dataSet)[0]
    # data assigned to cluster
    clusterAssment = np.mat(np.zeros((numberRow, 2)))
    cluster = [None] * dataSet.shape[0]
    # print cluster
    #initialized the centroids
    centroids = randomCenter(dataSet,k)
    clusterChanged = True
    while clusterChanged:

        clusterChanged = False
        for i in range(numberRow):
            # initial min to infinite
            minDist = inf
            minIndex = -1
            # find the centroid who is closest
            for j in range(k):
                #calculate the distance between each point and each centroid
                distance =sqrt(sum(power(centroids[j, :] - dataSet[i, :], 2)))
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            #update its cluster
            clusterAssment[i, :] = minIndex, minDist ** 2
            cluster[i] = minIndex

        # recalculate the new centroids based on cluster info
        for cent in range(k):
            pointInCluster = dataSet[nonzero(clusterAssment[:, 0].A == (cent))[0]]
            if len(pointInCluster):
                 centroids[(cent), :] = mean(pointInCluster, axis=0)

    return centroids,clusterAssment, cluster

def kmeansCost(centroids,data,k,cluster):
    cost = 0.0
    for i in range(len(data)):
        for j in range (k):
            cost += sqrt(sum(power(centroids[j, :] - data[i, :], 2)))
    return cost,clusters;



def hammingDistance(cluster,label):
    distance = 0.0
    n = len(label)
    for i in range(n):
        for j in range(n):
            if i != j:
                if label[i] == label[j] and cluster[i] != cluster[j]:
                    distance += 1
                if label[i] != label[j] and cluster[i] == cluster[j]:
                    distance+= 1
    return distance / (n * (n - 1))

def show(dataSet,clusterAssment):

    numSamples, dim = dataSet.shape
    mark =  ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr', 'xr', 'sb', 'sg', 'sk', '2r', '<b', '<g', '+b', '+g', 'pb']
    for i in xrange(numSamples):
       markIndex = int(clusterAssment[i, 0])
       plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
    plt.show()


if __name__=="__main__":

 # deleteCols = [4]  # [0,2,3,4,5,6,8,9,10,11,12,13]
 # data, label = loadDataset('iris.csv', deleteCols, 4)
 # costMin = inf
 # clustering = []
 # for i in range(100):
 #    centroids, clusterAssement, clusters = kmeans(data, 3)
 #    cost, cluster = kmeansCost(centroids, data, 3, clusters)
 #    if cost < costMin:
 #        costMin = cost
 #        clustering = cluster
 #
 # hammingDis = hammingDistance(label, clustering)
 # print "The hammingDistance of the first dataset : ", hammingDis


 # # ------------------------------Dataset 1----------------------------------#
 deleteCols = [0]#[0,2,3,4,5,6,8,9,10,11,12,13]
 data, label= loadDataset('wine.csv', deleteCols, 0)
 costMin = inf
 clustering = []
 for i in range(100):
  centroids,clusterAssement,clusters = kmeans(data,3)
  cost,cluster = kmeansCost(centroids,data,3,clusters)
  if cost < costMin:
      costMin = cost
      clustering = cluster

 hammingDis = hammingDistance(label, clustering)
 #print centroids
 #print clusterAssement
 #print clusters
 print "The hammingDistance of the first dataset : ", hammingDis
 #show(data,clusterAssement)
 # #  #------------------------------Dataset 2----------------------------------#
 deleteCols = [9]#[1,2,4,5,6,7,8,9]
 data, label = loadDataset('glass.csv', deleteCols, 9)
 costMin =inf
 clustering = []
 for i in range(100):
  centroids,clusterAssement,clusters = kmeans(data,3)
  cost,cluster = kmeansCost(centroids,data,3,clusters)
  if cost < costMin:
      costMin = cost
      clustering = cluster

 hammingDis = hammingDistance(label,clustering)
 #print clusters
 print "The hammingDistance of the Second dataset : ", hammingDis
 #show(data,clusterAssement)


