import numpy as np
import math
import copy


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

    def train(self):
        labels = copy.deepcopy(self.labels)
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
        bestFeat = self.getBestFest(dataSet)        # Return the best feature vector
        bestFeatLabel = labels[bestFeat]
        tree = {bestFeatLabel:{}}
        del(labels[bestFeat])                       # After picking up category of best feature, delete it from labels

