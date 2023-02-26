import unittest
import os

class ModelTest(unittest.TestCase):
    def test_readData(self):
        train = np.genfromtxt('data/MNIST_train_tiny.csv', delimiter=',')
        test = np.genfromtxt('data/MNIST_test_tiny.csv', delimiter=',')
        self.model.addTrainingData(train)
        self.model.addTestData(test)
        self.assertIsNotNone(self.model.train)
        self.assertIsNotNone(self.model.test)

    def test_addData(self):
        pass

    def test_classify(self):
        pass

    def test_loocv(self):
        pass