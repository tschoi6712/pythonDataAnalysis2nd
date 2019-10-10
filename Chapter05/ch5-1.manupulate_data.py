"""
데이터의 검색, 처리, 저장(1)
"""

## 1. Writing CSV files(넘파이와 판다스로 CSV 파일 쓰기
import numpy as np
import pandas as pd

# 난수 발생기로 3*4 넘파이 배열 만들기
np.random.seed(42)

a = np.random.randn(3, 4)
a[2][2] = np.nan
print(a)

path = 'D:\\1.Workspace\\1.Python\\part2.data_analysis\\pythonDataAnalysis2nd\\'

# np.savetxt()
np.savetxt(path + 'np.csv', a, fmt='%.2f', delimiter=',', header=" #1, #2,  #3,  #4")

# df.to_csv()
df = pd.DataFrame(a)
print(df)
df.to_csv(path + 'pd.csv', float_format='%.2f', na_rep="NAN!")

## 2. Comparing binary .npy format and pickle format(.npy와 피클 포맷)
import numpy as np
import pandas as pd
from tempfile import NamedTemporaryFile
from os.path import getsize

np.random.seed(42)
a = np.random.randn(365, 4)

# 임시파일로 저장하고 크기 확인
tmpf = NamedTemporaryFile()
np.savetxt(tmpf, a, delimiter=',')
print("Size CSV file", getsize(tmpf.name))

# .npy포맷으로 저장해 불러오고, 형태와 크기를 확인
tmpf = NamedTemporaryFile()
np.save(tmpf, a)
tmpf.seek(0)
loaded = np.load(tmpf)
print("Shape", loaded.shape)
print("Size .npy file", getsize(tmpf.name))

# 데이터프레임의 피클 포맷으로 만들고 읽어오기
df = pd.DataFrame(a)
print(df)
df.to_pickle('tmpf.pkl')
print("Size pickled dataframe", getsize('tmpf.pkl'))
print("DF from pickle\n", pd.read_pickle('tmpf.pkl'))

## 3. PyTables와 데이터 저장
# pip install numexpr tables - HDF5 라이브러리(데이터를 그룹과 데이터셋 형태로 구조화)

import numpy as np
import tables
from os.path import getsize

np.random.seed(42)
a = np.random.randn(365, 4)

# HDF5파일을 만들고 루트 노드에 배열을 붙인다
filename = 'pytable_demo.h5'
h5file = tables.open_file(filename, mode='w')
root = h5file.root
h5file.create_array(root, "array", a)
h5file.close()

# 파일을 읽고 크기를 출력
h5file = tables.open_file(filename, "r")
print(getsize(filename))

# 필요한 데이터를 찾기 위해 순회
for node in h5file.root:
    b = node.read()
    print(type(b), b.shape)

h5file.close()

## 4. Reading and writing DataFrames to HDF5(HDF5 로 데이터프레임을 읽기와 쓰기))
import numpy as np
import pandas as pd

np.random.seed(42)
a = np.random.randn(365, 4)

# 데모 파일에 HDFStore 생성자를 가져와 변수에 저장
filename = 'pytable_demo.h5'
store = pd.io.pytables.HDFStore(filename)
print(store)

# HDFStore에 데이터프레임을 저장
df = pd.DataFrame(a)
store['df'] = df
print(store)

# 데이터프레임에 접근 방법
print("Get", store.get('df').shape)
print("Lookup", store['df'].shape)
print("Dotted", store.df.shape)

# 저장된 요소를 삭제 - del, remove() 메서드
del store['df']
print("After del\n", store)

# 저장소가 열려 있는지 확인
print("Before close", store.is_open)
store.close()
print("After close", store.is_open)

# HDF 데이터를 쓰고 읽기
df.to_hdf('test.h5', 'data', format='table')
print(pd.read_hdf('test.h5', 'data', where=['index>363']))

## 5. Reading and writing to Excel(판다스로 엑셀 파일을 읽기와 쓰기)
# pip install openpyxl xlsxwriter xlrd

import numpy as np
import pandas as pd

np.random.seed(42)
a = np.random.randn(365, 4)

filename = 'excel_demo.xlsx'
df = pd.DataFrame(a)
print(filename)

df.to_excel(filename, sheet_name='Random Data')
print("Means\n", pd.read_excel(filename, 'Random Data').mean())

