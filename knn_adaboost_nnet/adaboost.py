#!/usr/bin/python

# Authors: Ayesha Bhimdiwala(aybhimdi), Umang Mehta(mehtau) & Vaishnavi Srinivasan(vsriniv)
# Please find the Report and Design Decisions listed in Report.pdf alongside.

import math
import operator
import random

import numpy as np


def buildDS(trainFile):
    global dsVector
    global dsOrient
    global numLinesTrain
    train = open(trainFile, "r")
    dsVector = np.zeros((numLinesTrain, 192), dtype=np.int_)
    dsOrient = np.zeros((numLinesTrain, 1), dtype=np.int_)
    for lineNumber, row in enumerate(train):
        rowList = row.split(' ', 2)
        dsOrient[lineNumber] = int(rowList[1])
        dsVector[lineNumber] = np.array([int(i) for i in rowList[2].split(' ')])
    train.close()


def buildTrain(orientation, trainWtList):
    global trainVector
    global trainOrient
    global featUdrCns
    global wtList
    global rdmRowIdx
    global numLinesTrain
    global dsVector
    global dsOrient
    rdmRowIdx = np.random.choice([i for i in range(numLinesTrain)], (numLinesTrain*75/100), replace=False, p=trainWtList.ravel())
    rdmRowIdx.sort()
    trainVector = np.zeros((len(rdmRowIdx), len(featUdrCns)), dtype=np.int_)
    trainOrient = np.zeros((len(rdmRowIdx), 1), dtype=np.int_)
    wtList = np.zeros((len(rdmRowIdx), 1), dtype=np.float_)
    for linenumber, row in enumerate(rdmRowIdx):
        trainOrient[linenumber] = 1 if dsOrient[row] == orientation else 0
        trainVector[linenumber] = [dsVector[row][col] for col in featUdrCns]
        wtList[linenumber] = trainWtList[row]


def decisionStump(vectorList):
    return 1 if vectorList[0] > vectorList[1] else 0


def adaboostTrain(trainFile, modelFile):
    modelAppend = open(modelFile, "w+")
    global numLinesTrain
    global rdmRowIdx
    global featUdrCns
    global wtList
    numLinesTrain = sum(1 for line in open(trainFile))
    trainWtList = np.ones((numLinesTrain, 1), dtype=np.float_) * (1.0 / numLinesTrain)
    buildDS(trainFile)
    for element in [0, 90, 180, 270]:
        for classifier in range(400):
            featUdrCns = random.sample(xrange(192), 2)
            featUdrCns.sort()
            buildTrain(element, trainWtList)
            pred = [1 if int(trainOrient[item]) == (decisionStump(trainVector[item])) else 0 for item in range(len(trainOrient))]
            alpha = (0.5) * math.log((pred.count(1) + 1) / float(pred.count(0) + 1))
            modelAppend.write("%s %s %s\n" % (str(element), ' '.join(str(i) for i in featUdrCns), str(alpha)))
            wtList = [math.exp(-alpha) if pred[el] == 1 else math.exp(alpha) for el in range(len(wtList))]
            wtList /= np.sum(wtList)
            for linenumber, row in enumerate(rdmRowIdx):
                trainWtList[row] = wtList[linenumber]
            trainWtList /= np.sum(trainWtList)
        # print "Classifier completed " + str(element)
    modelAppend.close()


def adaboostTest(testFile, modelFile):
    test = open(testFile, "r")
    output = open("output.txt", "w+")
    model = open(modelFile, "r")
    accuracy = 0
    numLinesTest = sum(1 for line in open(testFile))
    numLinesModel = sum(1 for line in open(modelFile))
    dsFleTst = np.empty(numLinesTest, dtype='S256')
    dsOntTst = np.zeros((numLinesTest, 1), dtype=np.int_)
    dsVtrTst = np.zeros((numLinesTest, 192), dtype=np.int_)
    dsOntMdl = np.zeros((numLinesModel, 1), dtype=np.int_)
    dsFtrMdl = np.zeros((numLinesModel, 2), dtype=np.int_)
    dsAphMdl = np.zeros((numLinesModel, 1), dtype=np.float_)
    for lineNumber, row in enumerate(test):
        rowList = row.split(' ', 2)
        dsFleTst[lineNumber] = rowList[0]
        dsOntTst[lineNumber] = int(rowList[1])
        dsVtrTst[lineNumber] = np.array([int(i) for i in rowList[2].split(' ')])
    test.close()
    for lineNumber, row in enumerate(model):
        rowList = row.split(' ')
        dsOntMdl[lineNumber] = int(rowList[0])
        dsFtrMdl[lineNumber] = [int(rowList[1]), int(rowList[2])]
        dsAphMdl[lineNumber] = float(rowList[3])
    test.close()
    for lineNumberT, rowT in enumerate(dsOntTst):
        pred = {0: 0, 90: 0, 180: 0, 270: 0}
        for lineNumberM, rowM in enumerate(dsOntMdl):
            orientVal = decisionStump([dsVtrTst[lineNumberT][col] for col in dsFtrMdl[lineNumberM]])
            if orientVal == 1:
                pred[int(rowM)] += float(dsAphMdl[lineNumberM])
            else:
                pred[int(rowM)] -= float(dsAphMdl[lineNumberM])
        predOnt = max(pred.iteritems(), key=operator.itemgetter(1))[0]
        output.write("%s %s\n" % (dsFleTst[lineNumberT], str(predOnt)))
        if predOnt == dsOntTst[lineNumberT]:
            accuracy += 1
    print "Adaboost Accuracy: " + str((accuracy/float(numLinesTest))*100)
