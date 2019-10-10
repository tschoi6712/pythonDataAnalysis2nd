"""
Pandas(1)
"""

## 1. pandas DataFrames - 관계형 데이터베이스의 일종인 2차원 자료구조
import numpy as np
import scipy as sp
import pandas as pd


# Load the data file into data frame
from pandas.io.parsers import read_csv

df = read_csv("ch3-1.WHO_first9cols.csv")                 # 데이터 파일을 데이터프레임 자료형으로 불러오기
print("Dataframe:\n", df)

print("Shape: ", df.shape)                          # 튜플 형상으로 변환
print()
print("Length: ", len(df))                          # 데이터프레임의 행 개수
print()
print("Column Headers:\n", df.columns)              # 열 이름
print()
print("Data types:\n", df.dtypes)                   # 열의 자료형
print()
print("Index:\n", df.index)                         # 데이터프레임의 인덱스(0부터 시작하소 행에서 1씩 증가)
print()
print("Values:\n", df.values)                       # 데이터프레임의 데이터 값(nan 빈 필드)


## 2. pandas Series - 라벨로 이뤄진 이형 1차원 자료 구조
country_col = df["Country"]
print("Type df: ", type(df))
print()
print("Type country col: ", type(country_col))      # 데이터프레임의 열은 시리즈 형임

print("Series shape: ", country_col.shape)
print()
print("Series index: ", country_col.index)
print()
print("Series values:\n", country_col.values)
print()
print("Series name: ", country_col.name)            # 시리즈 이름 = 데이터프레임의 열 이름

print("Last 2 countries: ", country_col[-2:])       # 시리즈의 마지막 2 행
print()
print("Last 2 countries type: ", type(country_col[-2:]))

# np.sign()에서 데이터프레임과 시리즈를 계산 - 양수의 반환 값은 1, 음수의 반환 값은 -1, 0은 숫자 0
last_col = df.columns[-1]
print("Last df column signs:\n", last_col, np.sign(df[last_col]), "\n")

# 넘파이 배열과 데이터프레임 간의 정렬 연산
np.sum([0, np.nan])
print(df.dtypes)
print(np.sum(df[last_col] - df[last_col].values))



## 3. Querying Data in pandas(데이터 검색)
import quandl

quandl.ApiConfig.api_key = "bZR-h4f5e4JLo3Zk6Ub2"           # quandl에서 api_key로 인증
sunspots = quandl.get("SIDC/SUNSPOTS_A")                    # sunspots 에는 데이터프레임이 저장

print("Head 2:\n", sunspots.head(2))                        # 처음과 마지막 두 개의 흑점 데이터
print("Tail 2:\n", sunspots.tail(2))

last_date = sunspots.index[-1]                              # 마지막 날짜의 마지막 값
print("Last value:\n", sunspots.loc[last_date])

print("Values slice by date:\n",
      sunspots["20020101": "20131231"])                     # 날짜 문자열로 검색

print("Slice from a list of indices:\n",
      sunspots.iloc[[2, 4, -4, -2]])                        # 인덱스를 사용해 원하는 구간을 검색

print("Scalar with Iloc:", sunspots.iloc[0, 0])             # 스칼라 값을 선택
print("Scalar with iat", sunspots.iat[1, 0])

print("Boolean selection:\n",
      sunspots[sunspots > sunspots.mean()])                 # 평균값보다 큰 값이 있는 열
print("Boolean selection with column label:\n",
      sunspots[sunspots['Number of Observations'] > sunspots['Number of Observations'].mean()])



## 4. Statistics with pandas DataFrame
import quandl

quandl.ApiConfig.api_key = "bZR-h4f5e4JLo3Zk6Ub2"           # quandl에서 api_key로 인증
sunspots = quandl.get("SIDC/SUNSPOTS_A")                    # sunspots 에는 데이터프레임이 저장

print("Describe", sunspots.describe(), "\n")                # 통계적으로 요약 설명
print("Non NaN observations", sunspots.count(), "\n")
print("MeanAbsoluteDeviation", sunspots.mad(), "\n")
print("Median", sunspots.median(), "\n")
print("Min", sunspots.min(), "\n")
print("Max", sunspots.max(), "\n")
print("Mode", sunspots.mode(), "\n")
print("Standard Deviation", sunspots.std(), "\n")
print("Variance", sunspots.var(), "\n")
print("Skewness", sunspots.skew(), "\n")
print("Kurtosis", sunspots.kurt(), "\n")






