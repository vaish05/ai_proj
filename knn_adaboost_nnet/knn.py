#!/usr/bin/python

# Authors: Ayesha Bhimdiwala(aybhimdi), Umang Mehta(mehtau) & Vaishnavi Srinivasan(vsriniv)
# Please find the Report and Design Decisions listed in Report.pdf alongside.

import math
from Queue import PriorityQueue

import numpy as np


def knnTrain(trainFile, modelFile):
    trainData = open(trainFile, "r")
    modelAppend = open(modelFile, "w")
    for line in trainData:
        row = line[:-1].split(' ', 2)
        modelAppend.write("%s|%s\n" % (row[1], row[2]))
    trainData.close()
    modelAppend.close()


def knnTest(testVector, trainOrient, trainVector, kValue, knn, knnDist):
    distQueue = PriorityQueue()
    for row in range(0,len(trainOrient),1):
        vector = trainVector[row]
        orient = int(trainOrient[row])
        eucDist = math.sqrt(np.sum(np.power((vector - testVector), 2)))
        distQueue.put((eucDist, orient))
    k=kValue
    for i in range(0, k, 1):
        knnScore = distQueue.get()
        knn[knnScore[1]] += 1
        knnDist[knnScore[1]] += knnScore[0]
    predictOrient = max(knn, key=knn.get)
    return predictOrient
