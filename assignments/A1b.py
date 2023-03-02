from src.semi_naive_knn import SemiNaiveKNN
import numpy as np
import pandas as pd
import os
import time
import sys

def main():
    train = np.genfromtxt('data/MNIST_train_small.csv', delimiter=',')

    #1b
    model = SemiNaiveKNN(1)
    model.addTrainingData(train)

    start = time.time()
    df = model.multi_loocv(0, range(1,21))
    end = time.time()

    print(f"LOOCV time: {end-start}")

    print()
    print(df)

    pd_df = pd.DataFrame(df, columns=["k", "loocv_err"])
    pd_df.to_csv("data/1b_output.csv")