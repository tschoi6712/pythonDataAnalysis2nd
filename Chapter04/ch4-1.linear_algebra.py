"""
통계학 및 선형대수(1)
"""

## 1. basic descriptive statistics(기본적인 통계학)
import numpy as np
from scipy.stats import scoreatpercentile
import pandas as pd


# statsmodels 데이터셋 패키지를 사용할 데이터 - 마우나로아의 이산화탄소 측정 값
path = 'D:\\1.Workspace\\1.Python\\part2.data_analysis\\pythonDataAnalysis2nd\\'
data = pd.read_csv(path + 'ch4-1.co2.csv', index_col=0, parse_dates=True)

co2 = np.array(data.co2)

print("The statistical values for amounts of co2 in atmosphere : \n")
print("Max method : ", co2.max())
print("Max function : ", np.max(co2))

print("Min method : ", co2.min())
print("Min function : ", np.min(co2))

print("Mean method : ", co2.mean())
print("Mean function : ", np.mean(co2))

print("Std method : ", co2.std())
print("Std function : ", np.std(co2))

print("Median : ", np.median(co2))
print("Score at percentile 50 : ", scoreatpercentile(co2, 50))



## 2. Linear Algebra(선형대수학)


# Inverting a matrix(역행렬) - 행렬 A * 역행렬 A-1 = 항등행렬 I
A = np.mat("2 4 6;4 2 6;10 -4 18")
print("A\n", A)

inverseA = np.linalg.inv(A)
print("inverse of A\n", inverseA)

print("Check\n", A * inverseA)                      # 항등행렬

print("Error\n", A * inverseA - np.eye(3))          # eye() - 단위행렬, zeros(), ones()


# Solving linear systems(선형방정식) - 행렬은 직선형의 한 벡터를 다른 벡터로 옮길 수 있다
A = np.mat("1 -2 1;0 2 -8;-4 5 9")
print("A\n", A)

b = np.array([0, 8, -9])
print("b\n", b)

# 선형시스템을 해석
x = np.linalg.solve(A, b)
print("Solution", x)

# 결과값을 확인
print("Check\n", np.dot(A, x))


## 3. Finding eigenvalues and eigenvectors(고유값과 고유벡터)
A = np.mat("3 -2;1 0")
print("A\n", A)

# 고유값을 계산 - A(이차원 행렬)x=a(일차원 벡터)x 방정식의 스칼라 해
print("Eigenvalues", np.linalg.eigvals(A))

# 고유값과 고유벡터 반환
eigenvalues, eigenvectors = np.linalg.eig(A)
print("First tuple of eig", eigenvalues)
print("Second tuple of eig\n", eigenvectors)

# 결과를 확인
for i in range(len(eigenvalues)):
    print("Left", np.dot(A, eigenvectors[:, i]))
    print("Right", eigenvalues[i] * eigenvectors[:, i])



## 4. random numbers(난수)

# binomial distribution(겜블링과 이항분포)
import numpy as np
from matplotlib.pyplot import plot, show

# 1000번의 동전 9개 뒤집기
cash = np.zeros(10000)
cash[0] = 1000
outcome = np.random.binomial(9, 0.5, size=len(cash))

# 5개보다 적은 동전에서 앞면이면 1개를 잃고 뒷면이면 1개를 얻는다
for i in range(1, len(cash)):
    if outcome[i] < 5:
        cash[i] = cash[i - 1] - 1
    elif outcome[i] < 10:
        cash[i] = cash[i - 1] + 1
    else:
        raise AssertionError("Unexpected outcome " + outcome)

print(outcome.min(), outcome.max())                     # 결과를 확인

plot(np.arange(len(cash)), cash)                        # 잔고는 랜덤워크 형태
show()


# normal distribution(정규분포 샘플링)
import numpy as np
import matplotlib.pyplot as plt

# 정규분포의 크기
N = 10000
normal_values = np.random.normal(size=N)

#히스토그램과 가우시안의 확률밀도함수
dummy, bins, dummy = plt.hist(normal_values, int(np.sqrt(N)), normed=True, lw=1)
sigma = 1
mu = 0
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu)**2 / (2 * sigma**2)), lw=2)

plt.show()


# normality test with scipy(정규성 검증)
import numpy as np
from scipy.stats import shapiro, anderson, normaltest


path = 'D:\\1.Workspace\\1.Python\\part2.data_analysis\\pythonDataAnalysis2nd\\'
flutrends = np.loadtxt(path + 'ch4-1.goog_flutrends.csv',
                       delimiter=',', usecols=(1,), skiprows=1,
                       converters={1: lambda s: float(s or 0)}, unpack=True)

N = len(flutrends)
normal_values = np.random.normal(size=N)
zero_values = np.zeros(N)

# Shapiro-Wilk 정규성 검정
print("Normal Values Shapiro", shapiro(normal_values))
print("Zeroes Shapiro", shapiro(zero_values))
print("Flu Shapiro", shapiro(flutrends))

# 앤더슨-달링 검정
print("Normal Values Anderson", anderson(normal_values))
print("Zeroes Anderson", anderson(zero_values))
print("Flu Anderson", anderson(flutrends))

# 디아고스티노-퍼슨 검정
print("Normal Values normaltest", normaltest(normal_values))
print("Zeroes normaltest", normaltest(zero_values))
print("Flu normaltest", normaltest(flutrends))




