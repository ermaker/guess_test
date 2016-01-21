from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.regularizers import l1l2, activity_l1l2

import json

class Model(object):
  def __init__(self):
    self.N = 5

  def load_data(self, filename):
    with open(filename) as f:
      self.data = json.loads(f.read())
    self.length = len(self.data) - self.N
    self.X = [self.data[idx:idx + self.N] for idx in xrange(self.length)]
    self.Y = self.data[self.N:]

  def train(self):
    self.model = Sequential()
    self.model.add(Dense(100, input_dim=self.N, W_regularizer=l1l2(), b_regularizer=l1l2(), activity_regularizer=activity_l1l2()))
    self.model.add(Activation('relu'))
    self.model.add(Dropout(0.25))
    self.model.add(Dense(100, W_regularizer=l1l2(), b_regularizer=l1l2(), activity_regularizer=activity_l1l2()))
    self.model.add(Activation('relu'))
    self.model.add(Dropout(0.25))
    self.model.add(Dense(100, W_regularizer=l1l2(), b_regularizer=l1l2(), activity_regularizer=activity_l1l2()))
    self.model.add(Activation('relu'))
    self.model.add(Dropout(0.25))
    self.model.add(Dense(100, W_regularizer=l1l2(), b_regularizer=l1l2(), activity_regularizer=activity_l1l2()))
    self.model.add(Activation('relu'))
    self.model.add(Dropout(0.25))
    self.model.add(Dense(1))
    self.model.compile(loss='mean_squared_error', optimizer='rmsprop')

    self.model.fit(np.array(self.X), np.array(self.Y), show_accuracy=True, batch_size=16, nb_epoch=2000, verbose=2)

  def predict(self, X):
    return self.model.predict(np.array([X]), verbose=0)[0][0]
