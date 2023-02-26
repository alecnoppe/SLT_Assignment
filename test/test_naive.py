import unittest
from test.model_test import ModelTest
from src.knn import KNearest

class TestNaive(ModelTest, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = KNearest(1)
    

if __name__ == '__main__':
    unittest.main()