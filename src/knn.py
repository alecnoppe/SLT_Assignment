import numpy as np
from math import sqrt

class KNearest:
    def __init__(self, k):
        self.K = k

    def addTrainingData(self, td):
        self.train = td
        self.num_rows, self.num_cols = self.train.shape

    def setK(self, k):
        self.K = k 

    def classifyTrain(self, dist_type):
        # split into folds
        # use 80% as train data
        # use 20% as test data
        pass

    def classifyTest(self, dist_type, test):
        # use training
        y_hat = []
        iteration = 0
        for test_row in test:
            neighbors = []
            for neighbor in range(0,self.K):
                min_distance = float('inf')
                next_neighbor = -1
                for row_index in range(0,self.num_rows):
                    if self.distance(dist_type, test_row, self.train[row_index]) < min_distance and row_index not in neighbors:
                        min_distance = self.distance(dist_type, test_row, self.train[row_index])
                        next_neighbor = row_index
                neighbors.append(next_neighbor)
            
            y_hat.append(self._knnRule(neighbors))

        return y_hat
            
    def _knnRule(self, neighbors):
        counts = [0,0,0, 0,0,0, 0,0,0]
        for n in neighbors:
            counts[int(self.train[n][0])-1]+=1
        
        max_value = 0
        index = 0

        for i, c in enumerate(counts):
            if c > max_value:
                max_value = c
                index = i
            
        return index+1
                
    
    def loocv(self, distance, test):
        pass
    
    def distance(self, dist_type, v, w,p=1):
        if dist_type==0:
            return self.euclideanDistance(v,w)
        elif dist_type==1:
            return self.minkowskiDistance(v,w,p)
        elif dist_type==2:
            return self.manhattanDistance(v,w)

    def euclideanDistance(self, v, w):
        # return sqrt ((w[1] - v[1])^2 + ... + (w[n-1] - v[n-1])^2)
        return sqrt(sum([(x[0]-x[1])**2 for x in zip(v[1:],w[1:])]))

    def minkowskiDistance(self, v, w, p=1):
        # return sqrt (| w[1] - v[1] |^p + ... + | w[n-1] - v[n-1] |^p)
        pass

    def manhattanDistance(self, v, w):
        # return | w[1] - v[1] | + ... + | w[n-1] - v[n-1] |
        pass
