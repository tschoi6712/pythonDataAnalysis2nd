"""
예측분석과 머신러닝 - (2) Logistic Regression(로지스틱 회귀분석을 이용한 분류)
"""


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import datasets
import numpy as np


# k-폴드 교차검증
def classify(x,  y):
    clf = LogisticRegression(random_state=12)
    scores = []
    kf = KFold(n_splits=10)
    for train, test in kf.split(x):
        clf.fit(x[train], y[train])
        scores.append(clf.score(x[test], y[test]))

    print("Accuracy: ", np.mean(scores))

rain = np.load('ch10.rain.npy')
dates = np.load('ch10.doy.npy')

x = np.vstack((dates[:-1], rain[:-1]))
y = np.sign(rain[1:])
classify(x.T, y)

#iris example
iris = datasets.load_iris()
x = iris.data[:, :2]
y = iris.target
classify(x, y)


