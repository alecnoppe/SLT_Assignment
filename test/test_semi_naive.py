import unittest
from model_test import ModelTest
from src.semi_naive_knn import SemiNaiveKNN

class TestSemiNaive(ModelTest):
    def __init__(self):
        self.model = SemiNaiveKNN(1)

if __name__ == '__main__':
    unittest.main()