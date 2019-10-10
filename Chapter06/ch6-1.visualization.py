"""
데이터 시각화(1)
"""

## 1. matplotlib 패키지 목록
import pkgutil as pu
import numpy as np
import matplotlib as mpl
import scipy as sp
import pandas as pd
import pydoc

print("Matplotlib version: ", mpl.__version__)


def clean(astr):
    s = astr
    # remove multiple spaces
    s = ' '.join(s.split())
    s = s.replace('=', '')

    return s


def print_desc(prefix, pkg_path):
    for pkg in pu.iter_modules(path=pkg_path):
        name = prefix + "." + pkg[1]

        if pkg[2]:
            try:
                docstr = pydoc.plain(pydoc.render_doc(name))
                docstr = clean(docstr)
                start = docstr.find("DESCRIPTION")
                docstr = docstr[start: start + 140]
                print(name, docstr)
            except:
                continue


print("\n")
print_desc("matplotlib", mpl.__path__)



## 2. Basic matplotlib plot
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 20)

plt.plot(x,  .5 + x)                                # 실선(기본값)
plt.plot(x, 1 + 2 * x, '--')                        # 점선

plt.show()



## 3. Logarithmic plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


path = 'D:\\1.Workspace\\1.Python\\part2.data_analysis\\pythonDataAnalysis2nd\\'

# 무어의 법칙의 관찰 데이터
df = pd.read_csv(path + 'ch6.transcount.csv')
df = df.groupby('year').aggregate(np.mean)
years = df.index.values
counts = df['trans_count'].values

# 데이터를 피팅 - 다항식 형태
poly = np.polyfit(years, np.log(counts), deg=1)
print("Poly", poly)

# 데이터를 플로팅하고 다항식을 평가
plt.semilogy(years, counts, 'o')
plt.semilogy(years, np.exp(np.polyval(poly, years)))

plt.show()



## 4. Scatter Plots
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

years = df.index.values
counts = df['trans_count'].values
gpu_counts = df['gpu_trans_count'].values
cnt_log = np.log(counts)

# c=칼러, s=버블 크기, alpha=투명도
plt.scatter(years, cnt_log, c=200 * years, s=20 + 200 * gpu_counts/gpu_counts.max(), alpha=0.5)

plt.show()



## 5. Legends and annotaions(범례와 주석)
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
years = df.index.values
counts = df['trans_count'].values
gpu_counts = df['gpu_trans_count'].values

poly = np.polyfit(years, np.log(counts), deg=1)
plt.plot(years, np.polyval(poly, years), label='Fit')

gpu_start = gpu.index.values.min()
y_ann = np.log(df.at[gpu_start, 'trans_count'])
ann_str = "First GPU\n %d" % gpu_start

# xy=해당하는 점, arrowprops=화살표 스타일, xytext=텍스트 위치, textcoords=Offset (in points) from the xy value
plt.annotate(ann_str, xy=(gpu_start, y_ann), arrowprops=dict(arrowstyle="->"), xytext=(-30, +70), textcoords='offset points')

cnt_log = np.log(counts)
plt.scatter(years, cnt_log, c=200 * years, s=20 + 200 * gpu_counts/gpu_counts.max(), alpha=0.5, label="Scatter Plot")
plt.legend(loc='upper left')
plt.grid()
plt.xlabel("Year")
plt.ylabel("Log Transistor Counts", fontsize=16)
plt.title("Moore's Law & Transistor Counts")

plt.show()























