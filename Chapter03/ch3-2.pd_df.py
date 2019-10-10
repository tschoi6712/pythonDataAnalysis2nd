"""
Pandas(2)
"""

## 5. Data Aggregation(데이터 수집)
import pandas as pd
from numpy.random import seed, rand, randint
import numpy as np


# 넘파이의 무작위 함수
seed(42)
df = pd.DataFrame({
                  'Weather': ['cold', 'hot', 'cold', 'hot',  'cold', 'hot', 'cold'],
                  'Food': ['soup', 'soup', 'icecream', 'chocolate',  'icecream', 'icecream', 'soup'],
                  'Price': 10 * rand(7),
                  'Number': randint(1, 9)
                  })
print(df)


# weather 열로 그룹화(2개 그룹)
weather_group = df.groupby('Weather')

i = 0
for name, group in weather_group:
      i = i + 1
      print("Group", i, name)
      print(group)

# 첫번째 열, 마지막 열, 각 그룹의 평균값
print("Weather group first\n", weather_group.first())
print("Weather group last\n", weather_group.last())
print("Weather group mean\n", weather_group.mean())


# 여러 열을 그룹화하고 새로운 그룹을 생성
wf_group = df.groupby(['Weather', 'Food'])
print("WF Groups", wf_group.groups)


# 넘파이 함수를 리스트 방식으로 적용
print("WF Aggregated\n", wf_group.agg([np.mean, np.median]))



## 6. Concatenating and appending DataFrames(행의 결합과 추가)

print("df :3\n", df[:3])                        # 첫번째 세 개의 열

print("Concat Back together\n", pd.concat([df[:3], df[3:]]))
print("Appending rows\n", df[:3].append(df[5:]))



## 7. joining DataFrames - 판다스.merge(), 데이터프레임.join()
path = 'D:\\1.Workspace\\1.Python\\part2.data_analysis\\pythonDataAnalysis2nd\\'

# 택시회사 직원번호와 목적지 데이터
dests = pd.read_csv(path + 'ch3-2.dest.csv')
print("Dests\n", dests)

# 택시운전기사의 팁 데이터
tips = pd.read_csv(path +'ch3-2.tips.csv')
print("Tips\n", tips)

print("Merge() on key\n", pd.merge(dests, tips, on='EmpNr'))
print("Dests join() tips\n", dests.join(tips, lsuffix='Dest', rsuffix='Tips'))

print("Inner join with merge()\n", pd.merge(dests, tips, how='inner'))
print("Outer join with merge()\n", pd.merge(dests, tips, how='outer'))



## 8. Handlng missing Values(누락된 데이터 다루기)
path = 'D:\\1.Workspace\\1.Python\\part2.data_analysis\\pythonDataAnalysis2nd\\'

df = pd.read_csv(path + 'ch3-1.WHO_first9cols.csv')

# country and Net primary school enrolment ratio male(%)의 헤더가 포함된 앞 세 개 열
df = df[['Country', df.columns[-2]]][:2]

print("New df\n", df)
print("Null Values\n", pd.isnull(df))
print("Total Null Values\n", pd.isnull(df).sum())           # Nan 값의 총 갯수(True = 1)
print("Not Null Values\n", df.notnull())
print("Last Column Doubled\n", 2 * df[df.columns[-1]])      # Nan 값을 곱하거나 더해도 Nan
print("Last Column plus NaN\n", df[df.columns[-1]] + np.nan)
print("Zero filled\n", df.fillna(0))



## 9. dealing with dates(날짜 다루기)
print("Date range", pd.date_range('1/1/1900', periods=42, freq='D'))


import sys
try:
    print("Date range", pd.date_range('1/1/1677', periods=4, freq='D'))
except:
    etype, value, _ = sys.exc_info()
    print("Error encountered", etype, value)

# pd.DateOffset으로 허용되는 날짜 범위
offset = pd.DateOffset(seconds=2 ** 33/10 ** 9)
mid = pd.to_datetime('1/1/1970')
print("Start valid range", mid - offset)
print("End valid range", mid + offset)

# 문자열을 날짜로 변환
print("With format", pd.to_datetime(['19021112', '19031230'], format='%Y%m%d'))

# 날짜가 아닌 문자열(둰째 열)은 변환되지 않음
print("Illegal date", pd.to_datetime(['1902-11-12', 'not a date']))
print("Illegal date coerced", pd.to_datetime(['1902-11-12', 'not a date'], errors='coerce'))



## 10. Pivot Tables
import pandas as pd
from numpy.random import seed, rand, randint
import numpy as np

seed(42)
N = 7
df = pd.DataFrame({
                    'Weather' : ['cold', 'hot', 'cold', 'hot', 'cold', 'hot', 'cold'],
                    'Food' : ['soup', 'soup', 'icecream', 'chocolate', 'icecream', 'icecream', 'soup'],
                    'Price' : 10 * rand(N), 'Number' : randint(1, 9)
})
print("DataFrame\n", df)
print(pd.pivot_table(df, columns=['Food'], aggfunc=np.sum))




