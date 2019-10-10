"""
예측분석과 머신러닝 - (1) PreProcessing(전처리)
"""


import numpy as np
from sklearn import preprocessing
from scipy.stats import anderson


# 네덜란드 기상청의 일별 강수량 - 평균, 분산, 앤더슨 달링 검정
rain = np.load('ch10.rain.npy')
rain = .1 * rain
rain[rain < 0] = .05/2
print("Rain mean", rain.mean())
print("Rain variance", rain.var())
print("Anderson rain", anderson(rain))

# 데이터의 스케일 조정
scaled = preprocessing.scale(rain)
print("Scaled mean", scaled.mean())
print("Scaled variance", scaled.var())
print("Anderson scaled", anderson(scaled))

# 기본 임계값=0
binarized = preprocessing.binarize(rain.reshape(-1, 1))
print(np.unique(binarized), binarized.sum())

# 정수를 라벨링
lb = preprocessing.LabelBinarizer()
lb.fit(rain.astype(int))
print(lb.classes_)






