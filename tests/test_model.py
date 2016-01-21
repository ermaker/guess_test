from guess import model
import numpy as np

import unittest

class ModelTest(unittest.TestCase):
  def test_load_data(self):
    m = model.Model()
    m.load_data('tests/fixtures/simple_sequence.json')
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

  def test_train_and_predict(self):
    X = [
      [1, 2, 3, 4, 5, 6],
      [2, 3, 4, 5, 6, 7],
      [3, 4, 5, 6, 7, 8],
      [4, 5, 6, 7, 8, 9],
      [5, 6, 7, 8, 9, 10]
    ]

    m = model.Model()
    m.load_data('data/sequence.json')
    m.train()
    print X[0], m.predict(X[0][:-1])
    print X[1], m.predict(X[1][:-1])
    print X[2], m.predict(X[2][:-1])
    print X[3], m.predict(X[3][:-1])
    print X[4], m.predict(X[4][:-1])
    V = [1,2,10,20,30,25]
    print V, m.predict(V[:-1])
    V = [23,23,22,21,21,21]
    print V, m.predict(V[:-1])
    V = [7,6,7,6,5,4]
    print V, m.predict(V[:-1])
    V = [10,9,8,7,6,5]
    print V, m.predict(V[:-1])
    V = [14,15,16,17,25,7]
    print V, m.predict(V[:-1])
    V = [100,101,102,103,104,105]
    print V, m.predict(V[:-1])
    V = [1000,1010,1020,1030,1040,1050]
    print V, m.predict(V[:-1])
    V = np.array([1000,1010,1020,1030,1040,1050]) * 10
    print V, m.predict(V[:-1])
    V = np.array(X[0]) * 10
    print V, m.predict(V[:-1])
    V = np.array(X[0]) * 100
    print V, m.predict(V[:-1])
    V = np.array(X[0]) * 1000
    print V, m.predict(V[:-1])
    V = np.array(X[1]) * 10
    print V, m.predict(V[:-1])
    V = np.array(X[2]) * 10
    print V, m.predict(V[:-1])
    V = np.array(X[3]) * 10
    print V, m.predict(V[:-1])
    V = np.array(X[4]) * 10
    print V, m.predict(V[:-1])

    actual = m.predict([1, 2, 3, 4, 5])
    self.assertEqual(6, actual)
