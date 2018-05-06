import pandas as pd
import numpy as np

def loadDataSet(filename,delecol,label):
    readFile =pd.read_csv(filename,sep=',',header=None,dtype=str,na_filter=False)
    labelCol =readFile.iloc[:,label].copy()
    attrCol = readFile.drop(readFile.columns[delecol], axis=1).copy()
    dataset = np.array(attrCol).astype('float')
    datasetToList = []
    for item in dataset:
        tmp =[]
        tmp.append(item)
        datasetToList.append(tmp)
    return dataset, datasetToList,np.array(labelCol).astype(np.str)

#calculate the minimum distance between two point
def minDistance(v1,v2):
    distance = float('inf')
    for i in v1:
        for j in v2:
            if distance > np.sqrt(np.sum(np.square(i - j))):
                distance = np.sqrt(np.sum(np.square(i - j)))
    return distance

#calculate the max distance between two point
def maxDistance(v1,v2):
    distance = 0.0
    for i in v1:
        for j in v2:
            if distance < np.sqrt(np.sum(np.square(i - j))):
                distance = np.sqrt(np.sum(np.square(i - j)))
    return distance
#calculate the average distance between two point
def avgDistance(v1,v2):
    distance = 0.0
    for i in v1:
        for j in v2:
            distance += np.sqrt(np.sum(np.square(i - j)))
    return distance / (len(v1) * len(v2))

def linkage(clusters, k,linkageWay):
    while len(clusters) > k:
        minpair = (-1, -1)
        mindis = float('inf')
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                #chooses which distance should be use
                if linkageWay =="single":
                    distance = minDistance(clusters[i],clusters[j])
                elif linkageWay =="complete":
                    distance = maxDistance(clusters[i], clusters[j])
                elif linkageWay =="average":
                    distance = avgDistance(clusters[i], clusters[j])
                #find the minimum distance between ith point to other
                if distance < mindis:
                    mindis = distance
                    minpair = (i, j)
        #get the most close point
        i, j = minpair
        #combine these two point
        clusters[i].extend(clusters[j])
        #since cluster[j] has cobined with cluster[i],so taht cluster[j] should be delete
        del clusters[j]
    return clusters

def assignLabel(data, clusters):
    n = len(data)
    label = np.zeros(n, dtype=np.int)  # track the nearest centroid
    for i in range(n):
        for j in range(len(clusters)):
            for d in clusters[j]:
                if (data[i] == d).all():
                    label[i] = j
                    # print "The label is: ",label[i]
    return label


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


if __name__=="__main__":
    #################################The first Dataset ########################################################
    #################################Test SingleLinkage ########################################################
    singleClusters = [[]]
    target = []
    deleteCol = [0]
    data, clusters, target = loadDataSet('wine.csv', deleteCol, 0)
    singleClusters =linkage(clusters, 3,'single');
    singlelabel = assignLabel(data, singleClusters)

    print "The hammingdistance of the first dataset(singleLinkage):   ", hammingDistance(singlelabel, target)


    #################################Test CompleteLinkage ########################################################
    data, clusters, target = loadDataSet('wine.csv', deleteCol, 0)
    completeCluster =[[]]
    completeCluster = linkage(clusters,3,'complete')
    completeLabel = assignLabel(data, completeCluster)
    print "The hammingdistance of the first dataset(completelinkage): ", hammingDistance(completeLabel, target)


    #################################Test AverageLinkage ########################################################
    data, clusters, target = loadDataSet('wine.csv', deleteCol, 0)
    avgCluster =[[]]
    avgCluster =linkage(clusters,3,'average')
    avgLabel= assignLabel(data,avgCluster)
    print "The hammingdistance of the first dataset(averageLinkage):  ", hammingDistance(avgLabel, target)



    #################################The Second Dataset ########################################################
    print
    print
    #################################Test SingleLinkage ########################################################
    singleClusters = [[]]
    target = []
    deleteCol = [9]
    data, clusters, target = loadDataSet('glass.csv', deleteCol, 9)
    singleClusters = linkage(clusters, 7, 'single');
    singlelabel = assignLabel(data, singleClusters)
    print "The hammingdistance of the second dataset(singleLinkage):  ", hammingDistance(singlelabel, target)
    #################################Test CompleteLinkage ########################################################
    data, clusters, target = loadDataSet('glass.csv', deleteCol, 9)
    completeCluster = [[]]
    completeCluster = linkage(clusters, 7, 'complete')
    completeLabel = assignLabel(data, completeCluster)
    print "The hammingdistance of the second dataset(completelinkage): ", hammingDistance(completeLabel, target)
    #################################Test AverageLinkage ########################################################
    data, clusters, target = loadDataSet('glass.csv', deleteCol, 9)
    avgCluster = [[]]
    avgCluster = linkage(clusters, 7, 'average')
    avgLabel = assignLabel(data, avgCluster)
    print "The hammingdistance of the second dataset(averageLinkage):  ", hammingDistance(avgLabel, target)













