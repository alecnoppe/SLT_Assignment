class KNearest:
    def __init__(self, k):
        self.K = k

    def addTrainingData(self, td):
        self.train = td
        self.size = 0 # tuples in train

    def setK(self, k):
        self.K = k 

    def classifyTrain(self, distance):
        # split into folds
        # use 80% as train data
        # use 20% as test data
        pass

    def classifyTest(self, distance, test):
        # use training
        pass
    
    def loocv(self, distance, test):
        pass

    def euclideanDistance(self, v, w):
        # return sqrt ((w[1] - v[1])^2 + ... + (w[n-1] - v[n-1])^2)
        pass

    def manhattanDistance(self, v, w):
        # return | w[1] - v[1] | + ... + | w[n-1] - v[n-1] |
        pass

    def minkowskiDistance(self, v, w, p=1):
        # return sqrt (| w[1] - v[1] |^p + ... + | w[n-1] - v[n-1] |^p)
        pass
