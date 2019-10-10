"""
Numpy 배열 다루기(1)
"""

import numpy as np

## 1. numpy arrays
a = np.arange(5)                # 배열 생성(벡터: 일차원 넘파이 배열)
print(a.dtype)                  # 배열의 데이터타입
print(a)
print(a.shape)                  # 배열 형상(튜플: 배열의 크기 값)


## 2. creating a multidimensional array
m = np.array([np.arange(2), np.arange(2)])      # 다차원 배열 생성
print(m)
print(m.shape)                                  # 다차원 배열 형상(2x2 행렬)


## 3. selecting numpy array elements
a = np.array([[1, 2], [3, 4]])                  # 2x2 배열 생성
print(a[0, 0])                                  # 배열 원소 추출하기([row, col])
print(a[0, 1])
print(a[1, 0])
print(a[1, 1])


## 4. numpy numerical types(숫자형)
np.float64(42)
np.int8(42.0)
np.bool(42)
np.bool(0)
np.float(True)
np.float(False)
np.arange(7, dtype=np.uint16)                   #  dtype=np.uint16 | 인수로 자료형 선택=부호가 없는 정수


# Data type objects(자료형 객체) - numpy.dtype 클래스의 인스턴스
print(a.dtype.itemsize)                         # 바이트 단위의 데이터 크기


# character codes(문자코드)
np.arange(7, dtype='f')                         # float
np.arange(7, dtype='D')                         # complex


# dtype constructors(생성자)
np.dtype(float)
np.dtype('f')                                   # float32
np.dtype('d')                                   # float64
np.dtype('f8')                                  # float64 : f바이트수
np.sctypeDict.keys()                            # 축약 형태의 문자코드



# dtype attribute
t = np.dtype('Float64')
print(t.char)                                   # 자료형의 문자코드
print(t.type)                                   # 배열원소의 객체 자료형
print(t.str)                                    # 해당 문자와 바이트 수



## 5. slicing and indexing(일차원 배열의 슬라이싱과 인덱싱)
a = np.arange(9)
print(a)
print(a[3:7])                                   # 인덱스 3부터 7까지
print(a[:7:2])                                  # 인덱스 0부터 7까지 2씩 증가
print(a[::-1])                                  # 역순으로 배열열



## 6. manipuating array shapes(배열 형상 다루기)
b = np.arange(24).reshape(2, 3, 4)              # 배열의 내부 데이터는 보존한 채로 형태만 변경 : reshape()
print(b)

print(b.ravel())                                # 다차원 배열을 1차원으로 변환(전 배열 형태로 환원)
print(b.flatten())                              # 다차원 배열을 1차원으로 변환(메모리를 새로 할당)

b.shape = (6, 4)                                # 튜플을 통해 형상을 구성하기
print(b)

print(b.transpose())                            # 전치행렬(행과 열의 데이터를 교환)

b.resize((2, 12))                               # 크기 재지정
print(b)


# stacking arrays(배열 합치기)
a = np.arange(9).reshape(3,3)
print(a)

b = 2 * a
print(b)

np.hstack((a, b))                               # 수평으로 합치기
np.concatenate((a, b), axis=1)

np.vstack((a, b))                               # 수직으로 합치기
np.concatenate((a, b), axis=0)

np.dstack((a, b))                               # 깊이 우선 합치기

oned = np.arange(2)
print(oned)
twice_oned = 2 * oned
print(twice_oned)

np.column_stack((oned, twice_oned))             # 열로 합치기
np.row_stack((oned, twice_oned))                # 행으로 합치기

np.column_stack((a, b))
np.column_stack((a, b)) == np.hstack((a, b))

np.row_stack((a, b))
np.row_stack((a,b)) == np.vstack((a, b))


# splitting arrays(배열 쪼개기)
a = np.arange(9).reshape(3, 3)
print(a)

np.hsplit(a, 3)                                 # 수평으로 쪼개기
np.split(a, 3, axis=1)

np.vsplit(a, 3)                                 # 수직으로 쪼개기
np.split(a, 3, axis=0)

c = np.arange(27).reshape(3, 3, 3)
print(c)
np.dsplit(c, 3)                                 # 깊이 우선 쪼개기(배열의 랭크는 3)


# numpy array attributes(배열 속성)
b = np.arange(24).reshape(2, 12)
print(b)

print(b.ndim)                                   # 배열의 차원 값
print(b.size)                                   # 원소의 갯수
print(b.itemsize)                               # 배열원소의 바이트 수
print(b.nbytes)                                 # 모든 원소의 바이트 수
print(b.size * b.itemsize)                      # 모든 원소의 바이트 수

b.resize(6, 4)
print(b)
print(b.T)                                      # 전치 = transpose()

b = np.array([1.j + 1, 2.j + 3])                # 복소수
print(b)
print(b.real)                                   # 실수 부분
print(b.imag)                                   # 허수 부분
print(b.dtype)                                  # 데이터타입
print(b.dtype.str)

b = np.arange(4).reshape(2, 2)
print(b)

f = b.flat                                      # numpy.flattier 객체를 반환
print(f)
for item in f:
    print(item)

print(b.flat[2])                                # flattier 객체의 원소에 접근
print(b.flat[[1,3]])

b.flat = 7                                      # 고정된 속성 값을 전체 배열의 값으로
print(b)

b.flat[[1, 3]] = 1                              # 고정된 속성 값을 선택된 배열의 값으로
print(b)

# converting arrays(배열 변환하기)
b = np.array([1.j + 1, 2.j + 3])
b.tolist()                                      # 리스트로 변환하기
b.astype(int)                                   # 원하는 자료형으로 변환
b.astype('complex')


## 7. creating array views and copies(배열 뷰 생성하고 복사하기)



