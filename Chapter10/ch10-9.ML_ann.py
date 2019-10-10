"""
예측분석과 머신러닝 - (9) Neural Networks: pip install theanets nose_parameterized
                                       conda install m2w64-toolchain
"""

import numpy as np
import theanets
import multiprocessing
from sklearn import datasets
from sklearn.metrics import accuracy_score


rain = .1 * np.load('ch10.rain.npy')
rain[rain < 0] = .05/2
dates = np.load('ch10.doy.npy')
x = np.vstack((dates[:-1], np.sign(rain[:-1])))
x = x.T

y = np.vstack(np.sign(rain[1:]),)
N = int(.9 * len(x))

train = [x[:N], y[:N]]
valid = [x[N:], y[N:]]

net = theanets.Regressor(layers=[2, 3, 1])

net.train(train, valid, learning_rate=0.1, momentum=0.5)

pred = net.predict(x[N:]).ravel()
print("Pred Min", pred.min(), "Max", pred.max())
print("Y Min", y.min(), "Max", y.max())
print("Accuracy", accuracy_score(y[N:], pred >= .5))

