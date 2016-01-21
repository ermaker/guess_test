from guess import model

import unittest

class ModelTest(unittest.TestCase):
  def test_load_data(self):
    m = model.Model()
    m.load_data()
    self.assertNotEqual(None, m.data)
