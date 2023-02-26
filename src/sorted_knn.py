from src.knn import KNearest

class SortedKNN(KNearest):
    def __init__(self, k):
        self.K = k

    def sortTrain(self):
        # idea : sort by most significant column?
        # idea : summarize many columns, 
        # idea : remove all empty columns
        self.train=self.train[self.train[:,1].argsort()]