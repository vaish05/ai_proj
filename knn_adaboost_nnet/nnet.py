#!/usr/bin/python

# Authors: Ayesha Bhimdiwala(aybhimdi), Umang Mehta(mehtau) & Vaishnavi Srinivasan(vsriniv)
# Please find the Report and Design Decisions listed in Report.pdf alongside.

import random
import numpy as np

opPositionVector = ['0', '90', '180', '270']
hiddenNeurons = 90
alpha = 0.0001
epoch = 1000


def sigmoid(x, derivative=False):
	return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


def train(trainFile, modelFile):
	trainData = open(trainFile, "r")
	# modelAppend = open(modelFile, "w")

	trainDataLength = sum(1 for l in trainData)
	trainData = open(trainFile, "r")

	trainIPVectors = np.zeros((trainDataLength,  192), dtype=np.int_)
	trainOPVectors = np.zeros((trainDataLength,  4), dtype=np.int8)
	for lineNo, line in enumerate(trainData):
		row = line[:-1].split(' ', 2)
		trainIPVectors[lineNo] = np.array(row[2].split(' '))
		trainOPVectors[lineNo, opPositionVector.index(row[1])] = 1

	np.random.seed(1)
	ipToHidden = np.random.uniform(-1, 1, size=(192, hiddenNeurons))
	hiddenToOP = np.random.uniform(-1, 1, size=(hiddenNeurons, 4))

	for i in range(epoch):
		vectorCheck = np.zeros(trainDataLength, dtype=np.bool_)
		while not np.all(vectorCheck):
			vectorIDX = random.randint(0, trainDataLength - 1)
			if vectorCheck[vectorIDX]:
				continue
			ipVector = trainIPVectors[vectorIDX]
			# Feed Forward
			hidden = np.dot(ipVector, ipToHidden)
			hiddenOP = sigmoid(hidden)

			op = np.dot(hiddenOP, hiddenToOP)
			finalOP = sigmoid(op)

			# Back-Propagation
			opGradient = sigmoid(np.asarray(finalOP), derivative=True)
			opError = opGradient * np.asarray(trainOPVectors[vectorIDX] - finalOP)
			hiddenGradient = sigmoid(np.asarray(hiddenOP), derivative=True)
			hiddenError = hiddenGradient * np.asarray(np.dot(opError, hiddenToOP.transpose()))

			hiddenToOP = hiddenToOP + alpha * np.dot(np.asmatrix(hiddenOP).transpose(), np.asmatrix(opError))
			ipToHidden = ipToHidden + alpha * np.dot(np.asmatrix(ipVector).transpose(), np.asmatrix(hiddenError))

			vectorCheck[vectorIDX] = True

		if i % 10 == 0:
			print "Epoch = ", i
			print "Epoch Error = ", epochError
			np.savez_compressed(modelFile, ipToHidden=ipToHidden, hiddenToOP=hiddenToOP)
			test("test-data.txt",  modelFile)
			print "-------------------------------------------------------------------------------------------------"
	np.savez_compressed(modelFile, ipToHidden=ipToHidden, hiddenToOP=hiddenToOP)
	print "Training Complete!!!"
	test("test-data.txt", modelFile)


def test(testFile, modelFile):
	modelData = np.load(modelFile)
	ipToHidden = modelData['ipToHidden']
	hiddenToOP = modelData['hiddenToOP']
	outputFile = open("output.txt", "w")

	numLinesTest = sum(1 for line in open(testFile))
	testVectors = np.zeros((numLinesTest, 192), dtype=np.int_)
	testOrients = np.zeros(numLinesTest, dtype=np.int_)
	imageNames = []
	lineNumber = 0
	for line in open(testFile, "r"):
		testList = line.split(' ')
		imageNames.append(testList[0])
		testList = [int(i) for i in testList[1:]]
		testOrients[lineNumber] = testList[0]
		testVectors[lineNumber] = np.array(testList[1:])
		lineNumber += 1

	correctCount = 0
	for idx, ipVector in enumerate(testVectors):
		# Feed Forward
		hidden = np.dot(ipVector, ipToHidden)
		hiddenOP = sigmoid(hidden)

		op = np.dot(hiddenOP, hiddenToOP)
		finalOP = sigmoid(op)
		predictedOrient = opPositionVector[np.argmax(finalOP)]
		outputFile.write("%s %s\n" % (str(imageNames[idx]), str(predictedOrient)))
		if int(predictedOrient) == testOrients[idx]:
			correctCount += 1
	print "Correctness: ", correctCount
	print "Accuracy: ", str(correctCount * 100.0 / numLinesTest)
	outputFile.close()
