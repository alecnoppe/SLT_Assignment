from src.knn import KNearest
import numpy as np
from math import sqrt

class SemiNaiveKNN(KNearest):
    def __init__(self, k):
        if k < 1:
            raise Exception('Choose k greater than or equal to 1!')
        self.K = k

    
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
            neighbors=np.array([self._distance(dist_type, test_row, w) for w in self.train]).argsort()[:self.K]
            
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
                neighbors=np.array([self._distance(dist_type, validation_row, w) for i, w in enumerate(self.train) 
                    if i!=validation_index]).argsort()[:self.K]
            
                y_hat.append(self._knnRule(neighbors))
            
        else:
            for test_row in self.test:
                neighbors=[self._distance(dist_type, test_row, w) for w in self.train].argsort()[:self.K]
            
                y_hat.append(self._knnRule(neighbors))

        return y_hat

    