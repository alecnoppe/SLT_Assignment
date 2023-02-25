from semi_naive_knn import SemiNaiveKNN
from knn import KNearest
import numpy as np
import os
import time


train = np.genfromtxt('data/MNIST_train_small.csv', delimiter=',')
test = np.genfromtxt('data/MNIST_test_tiny.csv', delimiter=',')

model = KNearest(3)
model.addTrainingData(train)
k = 3
n_test = 10

start = time.time()
y_hat = model.classifyTest(0, test[:n_test])
end = time.time()

print(f"Naive {k}-Nearest-Neighbors: on {n_test} entries: {end - start} seconds")
print()

model = SemiNaiveKNN(3)
model.addTrainingData(train)
k = 3
n_test = 10

start = time.time()
y_hat = model.classifyTest(0, test[:n_test])
end = time.time()

print(f"Semi-Naive {k}-Nearest-Neighbors: on {n_test} entries: {end - start} seconds")