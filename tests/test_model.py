from guess import model

import unittest

class ModelTest(unittest.TestCase):
  def test_load_data(self):
    m = model.Model()
    m.load_data()
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    self.assertEqual(data, m.data)

    self.assertEqual(5, m.N)
    self.assertEqual(5, m.length)


    X = [
      [1, 2, 3, 4, 5],
      [2, 3, 4, 5, 6],
      [3, 4, 5, 6, 7],
      [4, 5, 6, 7, 8],
      [5, 6, 7, 8, 9]
    ]
    Y = [
      6,
      7,
      8,
      9,
      10
    ]
    self.assertEqual(X, m.X)
    self.assertEqual(Y, m.Y)
