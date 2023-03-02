from src.semi_naive_knn import SemiNaiveKNN
import numpy as np
import pandas as pd
import os
import time
import sys

def main():
    train = np.genfromtxt('data/MNIST_train_small.csv', delimiter=',')
    test = np.genfromtxt('data/MNIST_test_small.csv', delimiter=',')

    #1a
    model = SemiNaiveKNN(1)
    model.addTrainingData(train)
    model.addTestData(test)

    errors = {"train_err": {}, "test_err": {}}
    df = []

    for i in range(1,21):
        errors["train_err"][i] = []
        errors["test_err"][i] = []

    start = time.time()
    for train_row in train:
        neighbors = model._getNeighbors(0, train_row)
        for k in range(1,21):
            model.setK(k)
            errors["train_err"][k].append(1 if model._knnRule(neighbors[:k]) == int(train_row[0]) else 0)
    end = time.time()
    print(f"Train time: {end-start}")

    start = time.time()
    for test_row in test:
        neighbors = model._getNeighbors(0, test_row)
        for k in range(1,21):
            model.setK(k)
            errors["test_err"][k].append(1 if model._knnRule(neighbors[:k]) == int(test_row[0]) else 0)
    end = time.time()
    print(f"Test time: {end-start} ")


    for k in range(1,21):
        df.append([k, np.mean(errors["train_err"][k]), np.mean(errors["test_err"][k])])

    print()
    print(df)

    pd_df = pd.DataFrame(df, columns=["k", "train_err", "test_err"])
    pd_df.to_csv("data/1a_output.csv")