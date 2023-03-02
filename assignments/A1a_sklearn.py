from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time

def main():
    train = np.genfromtxt('data/MNIST_train_small.csv', delimiter=',')
    test = np.genfromtxt('data/MNIST_test_small.csv', delimiter=',')

    df = []

    X_train = train[:,1:]
    Y_train = train[:,0]
    X_test = test[:,1:]
    Y_test = test[:,0]
    start = time.time()
    for k in range(1,21):
        model = KNeighborsClassifier(k)
        model.fit(X_train, Y_train)
        Y_hat = model.predict(X_train)
        Y_test_hat = model.predict(X_test)

        df.append([k, accuracy_score(Y_train,Y_hat), accuracy_score(Y_test,Y_test_hat)])



    end = time.time()

    print(f"Train/Test time: {end-start}")

    print(df)

    pd_df = pd.DataFrame(df, columns=["k", "train_err", "test_err"])
    pd_df.to_csv("1a_sklearn_output.csv")