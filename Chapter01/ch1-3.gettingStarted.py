"""
Matplotlib plots
"""

from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from matplotlib import pyplot as plt


iris = load_iris()
print(iris.DESCR)
data = iris.data
plt.plot(data[:, 0], data[:, 1], ".")
plt.show()


boston = load_boston()
print(boston.DESCR)
data = boston.data
plt.plot(data[:, 2], data[:, 4], "+")
plt.show()


