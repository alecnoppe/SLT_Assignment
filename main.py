from src.knn import KNearest
from src.semi_naive_knn import SemiNaiveKNN
import numpy as np

train = np.genfromtxt('data/MNIST_train_tiny.csv', delimiter=',')
test = np.genfromtxt('data/MNIST_test_tiny.csv', delimiter=',')
print(train[0][0])
print(test[0][0])

model = KNearest(1)
model.addTrainingData(train)
k = 1

while k <= 20:
    model.setK(k)
    model.classifyTrain(0)
    y_hat = model.classifyTest(0, test)
    total_wrong = 0
    for x in range(0,len(y_hat)):
        if model.train[x][0] != y_hat[x]:
            total_wrong+=1

    print((len(y_hat)-total_wrong)/len(y_hat))
    k+=1
# 1 a
    # kn = KNearest(1)
    # kn.addTrainingData(MNIST_train_small)
    # k = 1
    # do: 
    #   KNearest.setK(k)
    #   classify train(distance=euc)
    #   classify test(distance=euc, MNIST_test_small)
    # while k <= 20
# 1 b
# 1 c
    # kn = KNearest(1)
    # kn.addTrainingData(MNIST_train_small)
    # for k in range(1,21)
        # for p in range(1,16)
            # KNearest.setK(k)
            # classify train(distance=min, p=p)
            # classify test(distance=min, p=p)
# 1 d
    # euclidean vs minkowsky vs manhattan
    # normalization
    # 
# 1 e
# 1 f
# 1 g