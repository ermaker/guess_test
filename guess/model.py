from __future__ import absolute_import
from __future__ import print_function

import json

class Model(object):
  def __init__(self):
    self.N = 5

  def load_data(self):
    with open('data/sequence.json') as f:
      self.data = json.loads(f.read())
    self.length = len(self.data) - self.N
    self.X = [self.data[idx:idx + self.N] for idx in xrange(self.length)]
    self.Y = self.data[self.N:]
