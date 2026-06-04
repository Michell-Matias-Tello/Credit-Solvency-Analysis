import unittest
from src.data.loader import load_dataset

class TestDataLoader(unittest.TestCase):
    def test_load_dataset(self):
        data = load_dataset('data/credit_solvency_dataset.csv')
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)

if __name__ == '__main__':
    unittest.main()
