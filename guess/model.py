from __future__ import absolute_import
from __future__ import print_function

import json

class Model(object):
  def load_data(self):
    with open('data/sequence.json') as f:
      self.data = json.loads(f.read())
