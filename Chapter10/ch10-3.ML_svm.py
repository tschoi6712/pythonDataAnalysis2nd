"""
예측분석과 머신러닝 - (3) Support Vector Machines(서포트 벡터 머신을 이용한 분류)
"""


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from pprint import PrettyPrinter


# 그리드 검색(문제에 적절한 파라미터 찾기)
def classify(x, y):
    clf = GridSearchCV(SVC(random_state=42, max_iter=100), {'kernel': ['linear', 'poly', 'rbf'], 'C': [1, 10]})
    clf.fit(x, y)
    print("Accuracy: ", clf.score(x, y))
    PrettyPrinter().pprint(clf.cv_results_)

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




