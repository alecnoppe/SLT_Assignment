import unittest
from test.model_test import ModelTest
from src.semi_naive_knn import SemiNaiveKNN

class TestSemiNaive(ModelTest, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = SemiNaiveKNN(1)
        

if __name__ == '__main__':
    unittest.main()