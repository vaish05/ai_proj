#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: Ayesha Bhimdiwala(aybhimdi), Umang Mehta(mehtau) & Vaishnavi Srinivasan(vsriniv)
# (based on skeleton code by D. Crandall, Oct 2017)
#

# PD: character segment of a letter in the image; C: character
# Hidden states: Character
# Observable states: character segment of a letter in the image
##
## Simplified:
##-------------
## It is calculated using the formula: P(C|PD) = P(PD|C)P(C)/P(PD)
## Probability of character segment of a letter in the image is constant and it is ignored for calculating the simplified probability
## => P(C|PD) is directly proportional to P(PD|C)P(C)
## The maximum of the posterior probability for each character segment of a letter in the image is used to derive its letter.
## Character tags are independent of each other and the observed character segment of a letter in the image is dependent on the character tag.
##
## Variable Elimination:
##-----------------------
## References: https://www.cs.cmu.edu/~epxing/Class/10708-14/scribe_notes/scribe_note_lecture4.pdf
## 			   https://www.youtube.com/watch?v=7zDARfKVm7s
## We have used forward-backward algorithm for implementing variable elimination using HMM.
## Forward Matrix: stores the initial to final probability (P(C|PD)) score
## Backward Matrix: stores the final to initial probability (P(C|PD)) score
## For calculating the P(C|PD), each element in the forward matrix is multiplied with the corresponding element in the same cell of the backward matrix
## For each character segment of a letter in the image the maximum probability is calculated, and its respective letter is displayed.
## We formulate a HMM with characters as hidden state dependent on the previous character. Each observed character segment of a letter in the image is dependent fully on the corresponding character.
## 
## Viterbi:
##----------
## Calculated using the following Bellman Equation for Viterbi Decoding:
## v[t] = e(w[t]) {for i=1 to N max(v[t-1] * P(i,j))}
## where t is the current letter; t-1 is the previous letter and e() is the emission probability; v is posterior value; N is total number of characters in the training data
## We have taken transformation of this equation: v[t] = log(e(w[t])) {for i=1 to N max( log(v[t-1]) + log(P(i,j))}
## We have used two matrices
##	- store the value for posterior
##	- store the letter equivalent to {for i=1 to N argmax( log(v[t-1]) + log(P(i,j))}
## The formulation of HMM is same as that of HMM with Variable Elimination
##
## Emission:
##------------ 
## If the pixels match then we assign a value of 0.8 and else 0.2
## P(each character segment from the test image|training character) is the final product of the above for each pixel.
## 
## General Strategy discussion:
##----------------------------- 
## We had a discussion with group cgalani-sahk-skpanick for a general strategy to implement the respective algorithms
## 
## Output:
##--------
## Simplified: SUPREME COURT OF THF UN1TED STATES
## HMM VE: SUPREME COURT OF THF UN1TED STATES
## HMM MAP: SUPREME COURT OF THE UN1TED STATES
## 
#

from PIL import Image, ImageDraw, ImageFont
import sys
import numpy as np
import math

CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [["".join(['1' if px[x, y] < 1 else '0' for x in range(x_beg, x_beg + CHARACTER_WIDTH)]) for y in range(0, CHARACTER_HEIGHT)], ]
    return result


def load_training_letters(fname):
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return {TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS))}


#####
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
trainLetters = load_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
# print train_letters['a']
## Below is just some sample code to show you how the functions above work.
# You can delete them and put your own code here!

bitmaps = {}
letPtnTst = {}

characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
totalChar = 0
transitions = np.zeros((len(characters), len(characters)), dtype=np.int_)
charCount = {}
initial = {}
for c in characters:
    initial[c] = 0
    charCount[c] = 0

file = open(train_txt_fname, 'r');
for line in file:
    for ltIDX in range(len(line)):
        totalChar += 1
        if line[ltIDX] in characters:
            charCount[line[ltIDX]] += 1
        if ltIDX == 0 and (line[ltIDX] in characters):
            initial[line[ltIDX]] += 1
        if ltIDX > 0:
            if line[ltIDX - 1] not in characters or line[ltIDX] not in characters:
                ltIDX += 1
            else:
                transitions[characters.index(line[ltIDX - 1]), characters.index(line[ltIDX])] += 1

totalCharCount = sum(charCount.values())
totalInitials = sum(initial.values())

for i in characters:
    letter = train_letters[i]
    letterList = []
    for x in range(len(letter)):
        for y in range(len(letter[0])):
            letterList.append(letter[x][y])
    bitmaps[i] = letterList

pixelProb = np.zeros((CHARACTER_HEIGHT, CHARACTER_WIDTH), dtype=np.int_)

for bitmap in train_letters:
    for rowIDX in range(len(train_letters[bitmap])):
        for colIDX in range(len(train_letters[bitmap][rowIDX])):
            if train_letters[bitmap][rowIDX][colIDX] == '1':
                pixelProb[rowIDX, colIDX] += 1

def meanDensity(sentence):
    summation = 0
    for letter in sentence:
        for rowIDX in range(len(letter)):
            row = list(letter[rowIDX])
            for colIDX in range(len(row)):
                if row[colIDX] == '1':
                    summation += 1
    return summation / float(len(sentence)) 

def emissionCal(currentChar, flatSegment):
    matchAlpha = 1.0
    for idx in range(len(bitmaps[currentChar])):
        if bitmaps[currentChar][idx] == flatSegment[idx]:
            matchAlpha *= 0.8
        else:
            matchAlpha *= 0.2
    return matchAlpha
 
def simplified(sentence):
    charList = []
    for charSegment in sentence:
        flatSegment = [item for row in charSegment for item in row]
        maxProb = 0.0
        maxPOS = ''
        for currentChar in bitmaps:
            matchAlpha = 0
            for idx in range(len(bitmaps[currentChar])):
                if bitmaps[currentChar][idx] == flatSegment[idx]:
                    matchAlpha += 1
            simplifiedProb = matchAlpha / (14.0 * 25.0)
            if simplifiedProb > maxProb:
                maxProb = simplifiedProb
                maxPOS = currentChar

        charList.append(maxPOS)
    return charList


def hmm_ve(sentence):
    charList = []
    fwdState = np.zeros([len(characters), len(sentence)], np.float_)
    for charIDX, charSegment in enumerate(sentence):
        flatSegment = [item for row in charSegment for item in row]
        currentStateSum = 0.0
        for currentChar in characters:
            emission = emissionCal(currentChar, flatSegment)
            prevStateSum = 0
            if charIDX == 0:
                prevStateSum = 1.0 * initial[currentChar] / totalInitials
            else:
                for prevChar in characters:
                    transition = 1.0 * (transitions[characters.index(prevChar), characters.index(currentChar)] + 1) / (charCount[prevChar] + 1 if charCount[prevChar] > 0 else totalCharCount)
                    prevStateSum += transition
            fwdState[characters.index(currentChar), charIDX] = prevStateSum * emission
            currentStateSum += prevStateSum * emission
        for currentChar in characters:
            fwdState[characters.index(currentChar), charIDX] = fwdState[characters.index(currentChar), charIDX] / currentStateSum
            
    bwdState = np.zeros([len(characters), len(sentence)], np.float_)
    for charIDX in range(len(sentence) - 1, -1, -1):
        charSegment = sentence[charIDX]
        flatSegment = [item for row in charSegment for item in row]
        currentStateSum = 0.0
        for currentChar in characters:
            emission = emissionCal(currentChar, flatSegment)
            prevStateSum = 0
            if charIDX == 0:
                prevStateSum = 1.0 * charCount[currentChar] / totalInitials
            else:
                for prevChar in characters:
                    transition = 1.0 * (transitions[characters.index(prevChar), characters.index(currentChar)] + 1) / (charCount[prevChar] + 1 if charCount[prevChar] > 0 else totalCharCount)
                    prevStateSum += transition
            bwdState[characters.index(currentChar), charIDX] = prevStateSum * emission
            currentStateSum += prevStateSum * emission
        for currentChar in characters:
            fwdState[characters.index(currentChar), charIDX] = fwdState[characters.index(currentChar), charIDX] / currentStateSum

    convolution = np.multiply(fwdState, bwdState)
    return [characters[idx] for idx in np.argmax(convolution, axis=0)]

def hmm_viterbi(sentence):
    viterbi = np.empty([len(characters), len(sentence)], dtype=np.float_)
    maxPrevChar = np.empty([len(characters), len(sentence)], dtype='S4')
    for charIDX, charSegment in enumerate(sentence):
        for currentChar in characters:
            flatSegment = [item for row in charSegment for item in row]
            maxViterbi = - float('inf')
            maxChar = ''
            emissionProb = math.log(emissionCal(currentChar, flatSegment))
            if charIDX == 0:
                initialProb = math.log(initial[currentChar] + 1) - math.log(totalInitials + 1)
                matrixScore = emissionProb + initialProb
                viterbi[(characters.index(currentChar), charIDX)] = matrixScore
                maxPrevChar[(characters.index(currentChar), charIDX)] = ''
            else:
                for prevChar in characters:
                    transValue = (1.0/totalChar) if (transitions[characters.index(prevChar), characters.index(currentChar)]) <= 1 else (transitions[characters.index(prevChar), characters.index(currentChar)])
                    #transProb = math.log(transValue) - math.log(charCount[prevChar] if charCount[prevChar] > 0 else totalCharCount)
                    transProb = math.log(transValue)
                    interViterbi = (viterbi[characters.index(prevChar), charIDX - 1] + transProb)
                    if interViterbi > maxViterbi:
                        maxChar = prevChar
                        maxViterbi = interViterbi
                viterbi[(characters.index(currentChar), charIDX)] = maxViterbi + emissionProb
                maxPrevChar[(characters.index(currentChar), charIDX)] = maxChar

    lastChar = characters[np.argmax(viterbi, axis=0)[-1]]
    returnList = [lastChar]
    for charIDX in range(len(sentence) - 1, 0, -1):
        prevChar = maxPrevChar[characters.index(lastChar), charIDX]
        returnList.insert(0, prevChar)
        lastChar = prevChar
    return returnList


print "Simplified: " + "".join(simplified(test_letters))
print "HMM VE: " + "".join(hmm_ve(test_letters))
print "HMM MAP: " + "".join(hmm_viterbi(test_letters))

# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
# print ("\n".join([ r for r in train_letters['a'] ]))

# Same with test letters. Here's what the third letter of the test data
#  looks like:
# for cnt in range(len(test_letters)):
#    print ("\n".join([ r for r in test_letters[cnt] ]))
