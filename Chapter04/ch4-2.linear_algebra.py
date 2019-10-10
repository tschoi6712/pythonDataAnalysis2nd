"""
통계학 및 선형대수(2)
"""

## 5. Numpy masked array(마스킹된 배열)
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt


face = scipy.misc.face()

random_mask = np.random.randint(0, 2, size=face.shape)          # 마스크를 생성

plt.subplot(221)
plt.title("Original")
plt.imshow(face)
plt.axis('off')
masked_array = np.ma.array(face, mask=random_mask)              # 마스킹된 배열을 생성

plt.subplot(222)
plt.title("Masked")
plt.imshow(masked_array)
plt.axis('off')

plt.subplot(223)
plt.title("Log")
plt.imshow(np.ma.log(face).astype("float32"))
plt.axis('off')

plt.subplot(224)
plt.title("Log Masked")
plt.imshow(np.ma.log(masked_array).astype("float32"))
plt.axis('off')

plt.show()


## Disregarding negative and extreme values(음수 및 극단값들의 제거)
import numpy as np
from datetime import date
import sys
import matplotlib.pyplot as plt

path = 'D:\\1.Workspace\\1.Python\\part2.data_analysis\\pythonDataAnalysis2nd\\'

# MLB 선수들의 연봉 데이터
salary = np.loadtxt(path + 'ch4-2.MLB2008.csv', delimiter=',', usecols=(1,), skiprows=1, unpack=True)

# 3으로 나눠지는 숫자만 가지는 배열을 생성
triples = np.arange(0, len(salary), 3)
print("Triples", triples[:10], "...")

# 같은 크기의 연봉 정보를 담은 배열
signs = np.ones(len(salary))
print("Signs", signs[:10], "...")

# 세번째 원소를 음수화
signs[triples] = -1
print("Signs", signs[:10], "...")

# 로그(배열)
ma_log = np.ma.log(salary * signs)
print("Masked logs", ma_log[:10], "...")

# 극단값 제거
dev = salary.std()
avg = salary.mean()
inside = np.ma.masked_outside(salary, avg - dev, avg + dev)
print("Inside", inside[:10], "...")

plt.subplot(311)
plt.title("Original")
plt.plot(salary)

plt.subplot(312)
plt.title("Log Masked")
plt.plot(np.exp(ma_log))

plt.subplot(313)
plt.title("Not Extreme")
plt.plot(inside)

plt.subplots_adjust(hspace=.9)

plt.show()












