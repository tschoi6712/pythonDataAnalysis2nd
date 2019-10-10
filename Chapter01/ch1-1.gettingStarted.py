"""
파이썬과 numpy 의 배열 연산에 대한 경과시간 비교
The first vector to be added contains the squares of 0 up to n.
The second vector contains the cubes of 0 up to n.
The program prints the last 2 elements of the sum and the elapsed time
"""

import sys
from datetime import datetime
import numpy as np

def pythonsum(n):
    """
    벡터 덧셈 함수
    """
    a = list(range(n))
    b = list(range(n))
    c = []
    for i in range(len(a)):
        a[i] = i ** 2
        b[i] = i ** 3
        c.append(a[i] + b[i])
    return c
print(pythonsum(5))


def numpysum(n):
    """
    np.arange(n) 0 부터 n 까지의 정수 배열 생성
    """
    a = np.arange(n) ** 2
    b = np.arange(n) ** 3
    c = a + b
    return c
print(numpysum(5))

# 벡터의 원소 갯수
size = int(sys.argv[1])                         # sys.argv 프로그램을 실행시 입력된 값을 읽어 들일 수 옵션 값

start = datetime.now()
c = pythonsum(size)
delta = datetime.now() - start
print("The last 2 elements of the sum", c[-2:])
print("PythonSum elapsed time in microseconds", delta.microseconds)

start = datetime.now()
c = numpysum(size)
delta = datetime.now() - start
print("The last 2 elements of the sum", c[-2:])
print("NumPySum elapsed time in microseconds", delta.microseconds)




