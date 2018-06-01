#!/usr/bin/python

# Authors: Ayesha Bhimdiwala(aybhimdi), Umang Mehta(mehtau) & Vaishnavi Srinivasan(vsriniv)
# Please find the Report and Design Decisions listed in Report.pdf alongside.

import nnet
from adaboost import *
from knn_Predrag import *
import sys

def knnTestPreprocessor():
    lineNumber = 0
    model = open(modelFile, "r")
    for row in model:
        rowList = row.split('|')
        intmOrient = rowList[0]
        intmVector = rowList[1].split(' ')
        vector = [int(i) for i in intmVector]
        trainOrient[lineNumber] = intmOrient
        trainVector[lineNumber] = np.array(vector)
        lineNumber += 1
    model.close()
    lineNumber = 0
    testData = open(switchFile, "r")
    for line in testData:
        testList = line.split(' ')
        testFile[lineNumber] = testList[0]
        testList = [int(i) for i in testList[1:]]
        testOrient[lineNumber] = testList[0]
        testVector[lineNumber] = np.array(testList[1:])
        lineNumber += 1
    testData.close()

switch = sys.argv[1]
switchFile = sys.argv[2]
modelFile = sys.argv[3]
model = sys.argv[4]

if model.lower() == "nearest":
    if switch.lower() == "train":
        knnTrain(switchFile, modelFile)
    if switch.lower() == "test":
        output = open("output.txt", "w+")
        accuracy = 0
        numLinesTrain = sum(1 for line in open(modelFile))
        numLinesTest = sum(1 for line in open(switchFile))
        trainVector = np.zeros((numLinesTrain, 192), dtype=np.int_)
        trainOrient = np.zeros((numLinesTrain, 1), dtype=np.int_)
        testVector = np.zeros((numLinesTest, 192), dtype=np.int_)
        testOrient = np.zeros((numLinesTest, 1), dtype=np.int_)
        testFile = np.empty(numLinesTest, dtype='S256')
        knnTestPreprocessor()
        for row in range(0, len(testOrient), 1):
            knn = {0: 0, 90: 0, 180: 0, 270: 0}
            knnDist = {0: 0, 90: 0, 180: 0, 270: 0}
            # print "-------------LINE " + str(row) + "-------------"
            predictOrient = knnTest_Predrag(testVector[row], trainOrient, trainVector, 35, knn, knnDist, 2)
            output.write("%s %s\n" % (str(testFile[row]), str(predictOrient)))
            accuracy += (1 if predictOrient == int(testOrient[row]) else 0)
        print "K-Nearest Neighbours Accuracy: " + str(100.0 * accuracy / row)
        output.close()

elif model.lower() == "adaboost":
    if switch.lower() == "train":
        adaboostTrain(switchFile, modelFile)
    if switch.lower() == "test":
        adaboostTest(switchFile, modelFile)

elif model.lower() == "nnet":
    if switch.lower() == "train":
        nnet.train(switchFile, modelFile)
    if switch.lower() == "test":
        nnet.test(switchFile, modelFile)

elif model.lower() == "best":
    if switch.lower() == "train":
        nnet.train(switchFile, modelFile)
    if switch.lower() == "test":
        nnet.test(switchFile, modelFile)
