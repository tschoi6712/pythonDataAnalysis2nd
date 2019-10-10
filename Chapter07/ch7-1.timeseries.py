"""
Signal Processing and Time Series(신호처리와 시계열)(1)
"""

## 1. statsmodels 모듈
import pkgutil as pu
import numpy as np
import matplotlib as mpl
import scipy as sp
import pandas as pd
import pydoc
import statsmodels as sm
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 15, 6

print("Statsmodels version: ", sm.__version__)


def clean(astr):
    s = astr
    # remove multiple spaces
    s = ' '.join(s.split())
    s = s.replace('=', '')

    return s


def print_desc(prefix, pkg_path):
    for pkg in pu.iter_modules(path=pkg_path):
        name = prefix + "." + pkg[1]

        if pkg[2] == True:
            try:
                docstr = pydoc.plain(pydoc.render_doc(name))
                docstr = clean(docstr)
                start = docstr.find("DESCRIPTION")
                docstr = docstr[start: start + 140]
                print(name, docstr)
            except:
                continue


print("\n")
print_desc("statsmodels", sm.__path__)



## 2. 이동평균법(moving average)
import matplotlib.pyplot as plt
import statsmodels.api as sm
# from pandas.stats.moments import rolling_mean


# Load a dataset
data_loader = sm.datasets.sunspots.load_pandas()
df = data_loader.data

year_range = df["YEAR"].values
plt.plot(year_range, df["SUNACTIVITY"].values, label="Original")
plt.plot(year_range, df.rolling(window=11).mean()["SUNACTIVITY"].values, label="SMA 11")
plt.plot(year_range, df.rolling(window=22).mean()["SUNACTIVITY"].values, label="SMA 22")
plt.legend()
plt.show()



## 3. 윈도우 함수(window function)
import matplotlib.pyplot as plt
import statsmodels.api as sm
#from pandas import rolling_window
import pandas as pd

data_loader = sm.datasets.sunspots.load_pandas()
df = data_loader.data.tail(150)
df = pd.DataFrame({'SUNACTIVITY': df['SUNACTIVITY'].values}, index=df['YEAR'])
ax = df.plot()


def plot_window(win_type):
    df2 = df.rolling(22, win_type)
    df2.columns = [win_type]
    df2.plot(ax=ax)


plot_window('boxcar')
plot_window('triang')
plot_window('blackman')
plot_window('hanning')
plot_window('bartlett')
plt.show()



## 4. 공적분(Cointegration) - 두 개의 시계열 x(t)와 y(t)의 선형 결합이 일정
import statsmodels.api as sm
# from pandas.stats.moments import rolling_window
import pandas as pd
import statsmodels.tsa.stattools as ts
import numpy as np


def calc_adf(x, y):
    result = sm.OLS(x, y).fit()
    return ts.adfuller(result.resid)


data_loader = sm.datasets.sunspots.load_pandas()
data = data_loader.data.values
N = len(data)

# 사인파를 발생시키고 사인 자기 자신의 공적분을 계산
t = np.linspace(-2 * np.pi, 2 * np.pi, N)
sine = np.sin(np.sin(t))
print("Self ADF", calc_adf(sine, sine))

# 노이즈가 신호에 얼마나 영향을 주는지 확인
noise = np.random.normal(0, .01, N)
print("ADF sine with noise", calc_adf(sine, sine + noise))

# 진폭과 오프셋을 더 크게 한 코사인을 생성
cosine = 100 * np.cos(t) + 10
print("ADF sine vs cosine with noise", calc_adf(sine, cosine + noise))

# 사인과 흑점 사이의 공적분을 확인
print("Sine vs sunspots", calc_adf(sine, data))



## 5. 자기상관(autocorrelation)- 데이터셋에서 상관관계
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot


data_loader = sm.datasets.sunspots.load_pandas()
data = data_loader.data["SUNACTIVITY"].values

# 자기상관 값을 정규화
y = data - np.mean(data)
norm = np.sum(y ** 2)
correlated = np.correlate(y, y, mode='full') / norm
res = correlated[int(len(correlated) / 2):]

# 가장 큰 자기상관 계수를 찾기
print(np.argsort(res)[-5:])
plt.plot(res)
plt.grid(True)
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.show()
autocorrelation_plot(data)
plt.show()



## 6. 자기회귀모델(autoregressive) - 시계열 데이터에서 차후의 값을 예측
from scipy.optimize import leastsq
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np


def model(p, x1, x10):
    p1, p10 = p
    return p1 * x1 + p10 * x10


def error(p, data, x1, x10):
    return data - model(p, x1, x10)


def fit(data):
    p0 = [.5, 0.5]
    params = leastsq(error, p0, args=(data[10:], data[9:-1], data[:-10]))[0]
    return params


data_loader = sm.datasets.sunspots.load_pandas()
sunspots = data_loader.data["SUNACTIVITY"].values

cutoff = int(.9 * len(sunspots))
params = fit(sunspots[:cutoff])
print("Params", params)

pred = params[0] * sunspots[cutoff - 1:-1] + params[1] * sunspots[cutoff - 10:-10]
actual = sunspots[cutoff:]
print("Root mean square error", np.sqrt(np.mean((actual - pred) ** 2)))
print("Mean absolute error", np.mean(np.abs(actual - pred)))
print("Mean absolute percentage error", 100 * np.mean(np.abs(actual - pred) / actual))

mid = (actual + pred) / 2
print("Symmetric Mean absolute percentage error", 100 * np.mean(np.abs(actual - pred) / mid))
print("Coefficient of determination", 1 - ((actual - pred) ** 2).sum() / ((actual - actual.mean()) ** 2).sum())

year_range = data_loader.data["YEAR"].values[cutoff:]
plt.plot(year_range, actual, 'o', label="Sunspots")
plt.plot(year_range, pred, 'x', label="Prediction")
plt.grid(True)
plt.xlabel("YEAR")
plt.ylabel("SUNACTIVITY")
plt.legend()
plt.show()




