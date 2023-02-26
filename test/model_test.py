import unittest
import os
import numpy as np

class ModelTest(object):
    def test_readData(self):
        train = np.genfromtxt('data/MNIST_train_tiny.csv', delimiter=',')
        test = np.genfromtxt('data/MNIST_test_tiny.csv', delimiter=',')

        self.model.addTrainingData(train)
        self.model.addTestData(test)

        self.assertIsNotNone(self.model.train)
        self.assertIsNotNone(self.model.test)


    def test_classifyOdd(self):
        self.model.setK(1)
        y_hat = model.classifyTest(0, self.test[:10])
        self.assertEqual(y_hat, [7, 2, 1, 3, 9, 1, 4, 2, 6, 3])

        self.model.setK(3)
        y_hat = model.classifyTest(0, self.test[10:20])
        self.assertEqual(y_hat, [7, 4, 9, 0, 6, 7, 6, 7, 9, 4])

        self.model.setK(5)
        y_hat = model.classifyTest(0, self.test[20:30])
        self.assertEqual(y_hat, [0, 1, 3, 7, 0, 1, 1, 1, 5, 2])

        self.model.setK(7)
        y_hat = model.classifyTest(0, self.test[30:40])
        self.assertEqual(y_hat, [1, 2, 4, 0, 2, 8, 7, 8, 6, 1])


    def test_classifyEven(self):
        self.model.setK(2)
        y_hat = model.classifyTest(0, self.test[:10])
        self.assertEqual(y_hat, [7, 2, 1, 3, 9, 1, 4, 2, 6, 3])

        self.model.setK(4)
        y_hat = model.classifyTest(0, self.test[10:20])
        self.assertEqual(y_hat, [7, 4, 9, 0, 6, 7, 6, 7, 9, 4])

        self.model.setK(6)
        y_hat = model.classifyTest(0, self.test[20:30])
        self.assertEqual(y_hat, [0, 1, 3, 7, 0, 1, 1, 1, 5, 2])

        self.model.setK(8)
        y_hat = model.classifyTest(0, self.test[30:40])
        self.assertEqual(y_hat, [1, 2, 4, 0, 2, 8, 7, 8, 6, 1])


    def test_distances(self):
        self.model.setK(3)
        y_hat = model.classifyTest(0, self.test[:10])
        self.assertEqual(y_hat, [7, 2, 1, 3, 9, 1, 4, 2, 6, 3])

        y_hat_p2 = model.classifyTest(1, self.test[:10], 1)
        self.assertEqual(y_hat_p2, [7, 2, 1, 3, 9, 1, 4, 2, 6, 3])

        y_hat_p0 = model.classifyTest(1, self.test[:10], 2)
        self.assertEqual(y_hat_p0, [7, 2, 1, 3, 9, 1, 4, 2, 6, 3])

        y_hat_p3 = model.classifyTest(1, self.test[:10], 3)
        self.assertEqual(y_hat_p3, [7, 2, 1, 3, 9, 1, 4, 2, 6, 3])

        y_hat_m = model.classifyTest(2, self.test[:10])
        self.assertEqual(y_hat_m, [7, 2, 1, 3, 9, 1, 4, 2, 6, 3])

        self.assertEqual(y_hat, y_hat_p0)

        self.assertEqual(y_hat_m, y_hat_p2)

    def test_loocv(self):
        self.model.addTrainingData(self.model.train[:200])
        y_hat = self.model.loocv(0)
        self.assertEqual(y_hat, [1, 9, 6, 1, 1, 0, 1, 0, 1, 0, 1, 7, 1, 2, 0, 2, 7, 6, 1, 5, 3, 5, 1, 9, 1, 4, 7, 3, 4, 4, 5, 1, 3, 1, 1, 1, 4, 7, 8, 9, 9, 1, 0, 2, 6, 6, 1, 0, 9, 6, 6, 1, 1, 1, 1, 0, 2, 4, 1, 3, 0, 0, 4, 2, 0, 1, 0, 0, 9, 1, 9, 1, 7, 4, 6, 1, 0, 1, 7, 6, 1, 6, 0, 1, 2, 6, 1, 0, 2, 1, 1, 1, 1, 9, 1, 6, 5, 6, 6, 7, 2, 6, 0, 3, 2, 0, 7, 5, 0, 1, 1, 6, 6, 9, 2, 3, 2, 6, 2, 3, 2, 0, 1, 0, 0, 1, 7, 1, 3, 1, 5, 9, 4, 1, 6, 0, 8, 6, 7, 0, 1, 5, 1, 0, 4, 9, 1, 4, 1, 7, 1, 8, 3, 1, 7, 3, 7, 1, 1, 5, 1, 1, 3, 9, 3, 3, 1, 1, 3, 1, 6, 6, 3, 1, 1, 1, 4, 3, 0, 7, 7, 0, 7, 9, 7, 0, 3, 2, 4, 0, 1, 1, 7, 2, 2, 2, 9, 0, 7, 4])