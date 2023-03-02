import numpy as np
from math import sqrt
from scipy.spatial import distance

class KNearest:
    def __init__(self, k):
        if k < 1:
            raise Exception('Choose k greater than or equal to 1!')
        self.K = k

    def addTrainingData(self, td):
        self.train = td
        self.num_rows, self.num_cols = self.train.shape

    def addTestData(self, test):
        self.test = test
        self.num_trows, self.num_tcols = self.test.shape

    def setK(self, k):
        """
        set K after initialization

        :param k: k
        """
        self.K = k 

    def classifyTrain(self, dist_type):
        # split into folds
        # use 80% as train data
        # use 20% as test data
        pass

    def classifyTest(self, dist_type, test, p=1):
        """
        run KNN on test dataset

        :param test: numpy 2D array of vectors to be tested
        :param dist_type: type of distance to be used: 0 for euclidean, 1 for minkowski, 2 for manhattan
        :param p: optional parameter if chosen distance type is Minkowski distance
        :return: predicted values
        :rtype: int
        """
        y_hat = []
        
        for test_row in test:
            neighbors = []
            for neighbor in range(0,self.K):
                neighbors.append(self._nextNeighbor(dist_type, test_row, neighbors))
            
            y_hat.append(self._knnRule(neighbors))

        return y_hat
            
    def _nextNeighbor(self, dist_type, test_row, neighbors, p=1):
        """
        For internal use only. Returns closest neighbor (index) in 'self.train' to 'test_row' not in 'neighbors' according to 
        a given distance metric 'dist_type'.

        :param dist_type: type of distance to be used: 0 for euclidean, 1 for minkowski, 2 for manhattan
        :param p: optional parameter if chosen distance type is Minkowski distance
        :param test_row: vector to be labeled
        :param neighbors: already found indexes
        :return: next neighbor index
        :rtype: int
        """
        min_distance = float('inf')
        next_neighbor = -1
        for row_index in range(0,self.num_rows): #compute all distances once per search, sort, return first k rows
            if self._distance(dist_type, test_row, self.train[row_index]) < min_distance and row_index not in neighbors:
                min_distance = self._distance(dist_type, test_row, self.train[row_index])
                next_neighbor = row_index
        return next_neighbor
        
    def _knnRule(self, neighbors):
        """
        For internal use only. Returns label with most occurences in 'neighbors'
        :param neighbors: list of row indexes
        :return: label with most occurrences
        :rtype: int
        """
        counts = [0,0,0, 0,0,0, 0,0,0, 0]
        for n in neighbors:
            counts[int(self.train[n][0])]+=1
        
        max_value = 0
        index = 0

        for i, c in enumerate(counts):
            if c > max_value:
                max_value = c
                index = i
            if c == max_value:
                for x, y in enumerate(counts):
                    if y == max_value:
                        index = x
                        break
            
        return index
                
    
    def loocv(self, dist_type:int, p:int=1, test:bool=False):
        """
        Leave One Out Cross Validation. Classify each row in the training/set test data using the
        training set. Returns a list of classifications.
        :param dist_type: type of distance to be used: 0 for euclidean, 1 for minkowski, 2 for manhattan
        :param p: optional parameter if chosen distance type is Minkowski distance
        :param test: boolean to indicate which data to use (train=0 or test=1)
        :return: classifications of train/test data
        :rtype: list
        """
        y_hat = []

        if not test:
            for validation_index, validation_row in enumerate(self.train):
                neighbors = [validation_index]
                for neighbor in range(0,self.K):
                    neighbors.append(self._nextNeighbor(dist_type, validation_row, neighbors))
                
                y_hat.append(self._knnRule(neighbors[1:]))
            
        else:
            for test_row in self.test:
                neighbors = []
                for neighbor in range(0,self.K):
                    neighbors.append(self._nextNeighbor(dist_type, test_row, neighbors))
                
                y_hat.append(self._knnRule(neighbors))

        return y_hat
    
    def _distance(self, dist_type, v, w, p=1):
        """
        Distance hub for using different distance metrics.

        :param dist_type: type of distance to be used: 0 for euclidean, 1 for Minkowski, 2 for Manhattan
        :param v: test vector to be compared to all vectors from training set
        :param w: training vector to be compared
        :param p: optional parameter if chosen distance type is Minkowski distance
        :return: distance between vectors v and w
        :rtype: float
        """
        if dist_type==0:
            return self.euclideanDistance(v,w)
        elif dist_type==1:
            return self.minkowskiDistance(v,w,p)
        elif dist_type==2:
            return self.manhattanDistance(v,w)

    def euclideanDistance(self, v, w):
        """
        Calculates euclidean distance between vectors v and w. 
        d = sqrt((w[1] - v[1])^2 + ... + (w[n-1] - v[n-1])^2)

        :param v: numpy vector v
        :param w: numpy vector w
        :rtype: int
        """
        # return sqrt ((w[1] - v[1])^2 + ... + (w[n-1] - v[n-1])^2)
        # return sqrt(sum([(x[0]-x[1])**2 for x in zip(v[1:],w[1:])]))
        return np.linalg.norm(v[1:]-w[1:])

    def minkowskiDistance(self, v, w, p=1):
        """
        Calculates Minkowski distance between vectors v and w. 
        d = sqrt(| w[1] - v[1] |^p + ... + | w[n-1] - v[n-1] |^p)
        For Manhattan distance: use p=1
        For Euclidean distance: use p=2

        :param v: numpy vector v
        :param w: numpy vector w
        :param p: optional parameter if chosen distance type is Minkowski distance
        :rtype: int
        """
        # return sqrt (| w[1] - v[1] |^p + ... + | w[n-1] - v[n-1] |^p)
        # return sum([abs(x[0]-x[1])**p for x in zip(v[1:],w[1:])])**(1/p)
        return distance.minkowski(v[1:], w[1:], p)

    def manhattanDistance(self, v, w):
        """
        Calculates Manhattan distance between vectors v and w. 
        d = | w[1] - v[1] | + ... + | w[n-1] - v[n-1] |

        :param v: numpy vector v
        :param w: numpy vector w
        :rtype: int
        """
        # return | w[1] - v[1] | + ... + | w[n-1] - v[n-1] |
        # return sum([abs(x[0]-x[1]) for x in zip(v[1:],w[1:])])
        return distance.minkowski(v[1:], w[1:], 1)
