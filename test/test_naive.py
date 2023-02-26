import unittest
from model_test import ModelTest
from src.knn import KNearest

class TestNaive(ModelTest):
    def __init__(self):
        self.model = KNearest(1)
    
if __name__ == '__main__':
    unittest.main()