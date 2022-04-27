import math
from copy import deepcopy
import numpy as np


# import cPickle as Pickle

class ID3DTree(object):
    def __init__(self):
        self.tree = {}  # Store Trees
        self.dataSet = []  # Store DataSet
        self.labels = []  # Store LabelSet

    def loadDataSet(self, path, labels):
        recordlist = []
        # Load file content
        fp = open(path, "rb")
        content = fp.read()
        fp.close()

        # Change the content into 1D dataset
        rowlist = content.splitlines()
        recordlist = [row.split("\t") for row in rowlist if row.strip()]
        self.dataSet = recordlist
        self.labels = labels

    def loadDataSet2(self, data, labels):
        self.dataSet = data
        self.labels = labels

    def train(self):
        labels = deepcopy(self.labels)
        self.tree = self.buildTree(self.dataSet, labels)

    def maxCat(self, catelist):
        items = dict([(catelist.count(i), i) for i in catelist])
        return items[max(items.keys())]

    def buildTree(self, dataSet, labels):
        # Extract the category result(y data)
        cateList = [data[-1] for data in dataSet]

        # Termination condition 1: stop building tree and return the only category result
        # if there is only one kind of category result
        if cateList.count(cateList[0]) == len(cateList):
            return cateList[0]

        # Termination condition 2: stop building tree and return the category result
        # if there is only one category result
        if len(dataSet[0]) == 1:
            return self.maxCate(cateList)

        # Core
        bestFeat = self.getBestFest(dataSet)  # Return the best feature vector
        bestFeatLabel = labels[bestFeat]
        tree = {bestFeatLabel: {}}
        del (labels[bestFeat])  # After picking up category of best feature, delete it from labels

        uniqueVals = set([data[bestFeat] for data in dataSet])
        for value in uniqueVals:
            subLabels = labels[:]
            # seperate the dataset according to bestFeat & value
            splitDataset = self.splitDataSet(dataSet, bestFeat, value)
            subTree = self.buildTree(splitDataset, subLabels)        # build the subTree
            tree[bestFeatLabel][value] = subTree
        return tree

    def getBestFest(self, dataSet):
        # use this function to search the best feature vector
        # the last col of dataSet is y value, so numFeatures must minus one
        numFeatures = len(dataSet[0]) - 1

        # use y value to calculate the baseEntropy
        baseEntropy = self.computeEntropy(dataSet)
        bestInfoGain = 0.0  # initialize the bestInfoGain
        bestFeature = -1  # initialize the bestFeature

        # outside loop: check each col of dataSet and calculate the best Feature vector
        # take 'i' as an index for col of dataSet: range of 'i' is 0~(numFeature - 1 )
        # Ex: category: Age has three kinds of value young, middle-age and young
        # We calculate entropy for each values in each category
        for i in range(numFeatures):  # take the ith col of dataSet
            uniqueVals = set([data[i] for data in dataSet])  # remove duplicate values
            newEntropy = 0
            for value in uniqueVals:
                subDataSet = self.splitDataSet(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * self.computeEntropy(subDataSet)
            infoGain = baseEntropy - newEntropy
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature

    def computeEntropy(self, dataSet):
        # compute the Shannon Entropy
        data_len = float(len(dataSet))
        cateList = [data[-1] for data in dataSet]  # get the category result
        # set a dictionary ( key: category, value: times of occurrence
        items = dict([(i, cateList.count(i)) for i in cateList])
        infoEntropy = 0.0  # initialize the Shannon Entropy
        for key in items:  # Shannon Entropy: = -p*log2(p)
            prob = float(items[key]) / data_len  # infoEntropy = -prob * log(prob,2)
            infoEntropy -= prob * math.log(prob, 2)
        return infoEntropy

    # Delete the data row where the feature axis is located, and return the remaining data set
    def splitDataSet(self, dataSet, axis, value):
        rtnList = []
        for featVec in dataSet:
            if featVec[axis] == value:
                rFeatVec = featVec[:axis]
                rFeatVec = np.append(rFeatVec, featVec[axis + 1:])
                rtnList.append(rFeatVec)
        return rtnList

