"""
데이터 시각화(2)
"""

## 6. Three-dimensional plots
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


path = 'D:\\1.Workspace\\1.Python\\part2.data_analysis\\pythonDataAnalysis2nd\\'

df = pd.read_csv(path + 'ch6.transcount.csv')
df = df.groupby('year').aggregate(np.mean)
gpu = pd.read_csv(path + 'ch6.gpu_transcount.csv')
gpu = gpu.groupby('year').aggregate(np.mean)

df = pd.merge(df, gpu, how='outer', left_index=True, right_index=True)
df = df.replace(np.nan, 0)

fig = plt.figure()
ax = Axes3D(fig)
X = df.index.values
#Y = np.log(df['trans_count'].values)
Y = np.where(df['trans_count'].values > 0, np.ma.log(df['trans_count'].values), 0)

# 좌표행렬
X, Y = np.meshgrid(X, Y)
#Z = np.log(df['gpu_trans_count'].values)
Z = np.where(df['gpu_trans_count'].values > 0, np.ma.log(df['gpu_trans_count'].values), 0)

# Axes3D 클래스의 plot_surface 메서드로 데이터를 플로팅
surf = ax.plot_surface(X, Y, Z)

# API 메서드의 명명 규칙 set_함수이름
ax.set_xlabel('Year')
ax.set_ylabel('Log CPU transistor counts')
ax.set_zlabel('Log GPU transistor counts')
ax.set_title("Moore's Law & Transistor Counts")
plt.show()



## 7. plotting in pandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


path = 'D:\\1.Workspace\\1.Python\\part2.data_analysis\\pythonDataAnalysis2nd\\'

df = pd.read_csv(path + 'ch6.transcount.csv')
df = df.groupby('year').aggregate(np.mean)
gpu = pd.read_csv(path + 'ch6.gpu_transcount.csv')
gpu = gpu.groupby('year').aggregate(np.mean)

df = pd.merge(df, gpu, how='outer', left_index=True, right_index=True)
df = df.replace(np.nan, 0)

df.plot()
df.plot(logy=True)

# kind=차트 종류, loglog=로그 로그 그래프
df[df['gpu_trans_count'] > 0].plot(kind='scatter', x='trans_count', y='gpu_trans_count', loglog=True)
plt.show()



## 8. Lag plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import lag_plot


path = 'D:\\1.Workspace\\1.Python\\part2.data_analysis\\pythonDataAnalysis2nd\\'

df = pd.read_csv(path + 'ch6.transcount.csv')
df = df.groupby('year').aggregate(np.mean)
gpu = pd.read_csv(path + 'ch6.gpu_transcount.csv')
gpu = gpu.groupby('year').aggregate(np.mean)

df = pd.merge(df, gpu, how='outer', left_index=True, right_index=True)
df = df.replace(np.nan, 0)

# 시계열 데이터 형식의 지연 plot(기본값 1)
lag_plot(np.log(df['trans_count']))
plt.show()



## 9. autocorrelation plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot


path = 'D:\\1.Workspace\\1.Python\\part2.data_analysis\\pythonDataAnalysis2nd\\'

df = pd.read_csv(path + 'ch6.transcount.csv')
df = df.groupby('year').aggregate(np.mean)
gpu = pd.read_csv(path + 'ch6.gpu_transcount.csv')
gpu = gpu.groupby('year').aggregate(np.mean)

df = pd.merge(df, gpu, how='outer', left_index=True, right_index=True)
df = df.replace(np.nan, 0)

# 자기상관(시간 지연이 1일 때 n과 n+1 시간 간의 상관관계)
autocorrelation_plot(np.log(df['trans_count']))
plt.show()



## 10. plot.ly(온라인 데이터 시각화 도구) ???
#pip install plotly

import plotly.plotly as py
from plotly.graph_objs import *
import numpy as np
import pandas as pd


path = 'D:\\1.Workspace\\1.Python\\part2.data_analysis\\pythonDataAnalysis2nd\\'

df = pd.read_csv(path + 'ch6.transcount.csv')
df = df.groupby('year').aggregate(np.mean)
gpu = pd.read_csv(path + 'ch6.gpu_transcount.csv')
gpu = gpu.groupby('year').aggregate(np.mean)

df = pd.merge(df, gpu, how='outer', left_index=True, right_index=True)
df = df.replace(np.nan, 0)

# Change the user and api_key to your own username and api_key
py.sign_in('username', 'api_key')

counts = np.log(df['trans_count'].values)
gpu_counts = np.log(df['gpu_trans_count'].values)


data = Data([Box(y=counts), Box(y=gpu_counts)])
plot_url = py.plot(data, filename='moore-law-scatter')
print(plot_url)

















