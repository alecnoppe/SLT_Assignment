from src.semi_naive_knn import SemiNaiveKNN
from src.knn import KNearest
import numpy as np
import os
import time
import sys


train = np.genfromtxt('data/MNIST_train_small.csv', delimiter=',')
test = np.genfromtxt('data/MNIST_test_tiny.csv', delimiter=',')

if __name__ == "__main__" and len(sys.argv) > 2:
    args = sys.argv
    k = int(args[1])
    n_test = int(args[2])
elif __name__ == "__main__" and len(sys.argv) > 1:
    args = sys.argv
    k = int(args[1])
    n_x, n_test = test.shape
else:
    k = 3
    n_test = 10

model = SemiNaiveKNN(k)
model.addTrainingData(train[:200])

start = time.time()
y_hat = model.loocv(0)
print(y_hat)
end = time.time()

print(f"Semi-Naive {k}-Nearest-Neighbors: on {n_test} entries: {end - start} seconds")