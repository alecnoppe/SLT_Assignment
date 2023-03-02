from src.knn import KNearest
import numpy as np
from math import sqrt
import time

class SemiNaiveKNN(KNearest):
    def __init__(self, k):
        if k < 1:
            raise Exception('Choose k greater than or equal to 1!')
        self.K = k
        self.neighbor_time = 0
        self.neighbor_sort_time = 0
        self.neighbor_get_time = 0
        self.rule_time = 0

    def _knnRule(self, neighbors):
        """
        For internal use only. Returns label with most occurences in 'neighbors'
        :param neighbors: list of row indexes
        :return: label with most occurrences
        :rtype: int
        """
        start = time.time()
        counts = [[0],[0],[0], [0],[0],[0], [0],[0],[0], [0]]

        for train_row in neighbors:
            counts[int(train_row[1])][0]+=1
            counts[int(train_row[1])].append(train_row[0])
        
        max_value = 0
        index = 0

        for i, c in enumerate(counts):
            if c[0] > max_value:
                max_value = c[0]
                index = i
            elif c[0] == max_value != 0:
                avg_new = np.mean(counts[i][1:])
                avg_max = np.mean(counts[index][1:])

                if avg_new < avg_max:
                    max_value = c[0]
                    index = i
                    
        end = time.time()
        self.rule_time += end-start
        return index
    
    def _getNeighbors(self, dist_type, test_row, p=1):
        """
        For internal use only. Calculate neighbors with shortest distance to test_row.
        :param test: numpy 2D array of vectors to be tested
        :param dist_type: type of distance to be used: 0 for euclidean, 1 for minkowski, 2 for manhattan
        :param p: optional parameter if chosen distance type is Minkowski distance
        :return: sorted neighbors to test_row
        :rtype: numpy array
        """
        start = time.time()
        neighbors = np.array([[self._distance(dist_type, test_row, w, p), int(w[0])] for w in self.train])
        mid = time.time()
        self.neighbor_get_time += mid - start
        neighbors = neighbors[neighbors[:,0].argsort()]
        end = time.time()
        self.neighbor_sort_time += end - mid
        self.neighbor_time += end-start
        return neighbors


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
            neighbors=self._getNeighbors(dist_type, test_row, p)[:self.K]
            y_hat.append(self._knnRule(neighbors))

        return y_hat

    


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
                neighbors=np.array([[self._distance(dist_type, validation_row, w, p), int(w[0])] for i, w in enumerate(self.train) 
                    if i!=validation_index])
                
                neighbors = neighbors[neighbors[:,0].argsort()][:self.K]
            
                y_hat.append(self._knnRule(neighbors))
            
        else:
            for test_row in self.test:
                neighbors=[[self._distance(dist_type, test_row, w, p), int(w[0])] for w in self.train]
                neighbors = neighbors[neighbors[:,0].argsort()][:self.K]
                y_hat.append(self._knnRule(neighbors))

        return y_hat
    
    def multi_loocv(self, dist_type:int, k_range:list, p:int=1):
        """
        Leave One Out Cross Validation. Classify each row in the training/set test data using the
        training set. Returns a list of classifications.
        :param dist_type: type of distance to be used: 0 for euclidean, 1 for minkowski, 2 for manhattan
        :param p: optional parameter if chosen distance type is Minkowski distance
        :param test: boolean to indicate which data to use (train=0 or test=1)
        :return: classifications of train/test data
        :rtype: list
        """
        errors = {"loocv_err": {}}
        for k in k_range:
            errors['loocv_err'][k]=[]

        for validation_index, validation_row in enumerate(self.train):
            neighbors=np.array([[self._distance(dist_type, validation_row, w, p), int(w[0])] for i, w in enumerate(self.train) 
                if i!=validation_index])
            neighbors = neighbors[neighbors[:,0].argsort()]

            for k in k_range:
                errors['loocv_err'][k].append(1 if self._knnRule(neighbors[:k]) == int(validation_row[0]) else 0)
        
        df = []
        for k in range(1,21):
            df.append([k, np.mean(errors["loocv_err"][k])])
        return df
    
    def multi_loocv_p(self, dist_type:int, k_range:list, p_range:list):
        """
        Leave One Out Cross Validation. Classify each row in the training/set test data using the
        training set. Returns a list of classifications.
        :param dist_type: type of distance to be used: 0 for euclidean, 1 for minkowski, 2 for manhattan
        :param p: optional parameter if chosen distance type is Minkowski distance
        :param test: boolean to indicate which data to use (train=0 or test=1)
        :return: classifications of train/test data
        :rtype: list
        """
        errors = {"loocv_err": {}}
        for k in k_range:
            errors['loocv_err'][k]={}
            for p in p_range:
                errors['loocv_err'][k][p]=[]

        for validation_index, validation_row in enumerate(self.train):
            if validation_index % 100==0:
                print(validation_index)
            for p in p_range:
                neighbors=np.array([[self._distance(dist_type, validation_row, w, p), int(w[0])] for i, w in enumerate(self.train) 
                    if i!=validation_index])
                neighbors = neighbors[neighbors[:,0].argsort()]

                for k in k_range:
                    errors['loocv_err'][k][p].append(1 if self._knnRule(neighbors[:k]) == int(validation_row[0]) else 0)
        
        df = []
        for k in range(1,21):
            for p in p_range:
                df.append([k,p, np.mean(errors["loocv_err"][k][p])])
        return df


    