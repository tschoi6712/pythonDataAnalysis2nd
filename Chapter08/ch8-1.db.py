"""
데이터베이스로 작업하기(1)
"""


## 1. sqlite3
import sqlite3

with sqlite3.connect(":memory:") as con:

    # db 커서 만들기
    c = con.cursor()

    # 테이블 생성
    c.execute('''CREATE TABLE sensors (date text, city text, code text, sensor_id real, temperature real)''')

    # 데이터 접근
    for table in c.execute("SELECT name FROM sqlite_master WHERE type = 'table'"):
        print("Table", table[0])

    # 데이터 삽입과 조회
    c.execute("INSERT INTO sensors VALUES ('2016-11-05','Utrecht','Red',42,15.14)")
    c.execute("SELECT * FROM sensors")

    # 데이터 가져오기
    print(c.fetchone())

    # 테이블 삭제 후, 테이블 갯수를 출력
    con.execute("DROP TABLE sensors")
    print("# of tables", c.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'").fetchone()[0])

    c.close()



## 2. pandas로 데이터베이스 접근하기
import statsmodels.api as sm
from pandas.io.sql import read_sql
import sqlite3


with sqlite3.connect(":memory:") as con:
    c = con.cursor()

    data_loader = sm.datasets.sunspots.load_pandas()
    df = data_loader.data
    rows = [tuple(x) for x in df.values]

    con.execute("CREATE TABLE sunspots(year, sunactivity)")
    # 튜플 리스트에 데이터 삽입
    con.executemany("INSERT INTO sunspots(year, sunactivity) VALUES (?, ?)", rows)
    c.execute("SELECT COUNT(*) FROM sunspots")
    print(c.fetchone())
    print("Deleted", con.execute("DELETE FROM sunspots where sunactivity > 20").rowcount, "rows")

    # 데이터프레임으로 반환
    print(read_sql("SELECT * FROM sunspots where year < 1732", con))
    con.execute("DROP TABLE sunspots")

    c.close()


## 4. Pony ORM
#pip install pony
from pony.orm import Database, db_session
import statsmodels.api as sm


db = Database('sqlite', ':memory:')

with db_session:
    data_loader = sm.datasets.sunspots.load_pandas()
    df = data_loader.data
    df.to_sql("sunspots", db.get_connection())
    print(db.select("count(*) FROM sunspots"))



## 5. dataset: 사용하기 쉬운 데이터베이스
#pip install dataset
import dataset


db = dataset.connect('sqlite:///:memory:')

# 테이블 생성
table = db["books"]
table.insert(dict(title="NumPy Beginner's Guide", author='Ivan Idris'))
table.insert(dict(title="NumPy Cookbook", author='Ivan Idris'))
table.insert(dict(title="Learning NumPy", author='Ivan Idris'))

for row in db["books"]:
    print(row)

print("Tables", db.tables)



## 6. pymongo와 mongodb - NoSQL 도큐먼트 지향 데이터베이스
#pip install pymongo
#MongoDB Community Server 버전 다운로드

from pymongo import MongoClient
import statsmodels.api as sm
import json
import pandas as pd

client = MongoClient()
db = client.test_database

# 데이터프레임의 JSON 파일을 생성하고 mongodb에 저장
data_loader = sm.datasets.sunspots.load_pandas()
df = data_loader.data
rows = json.loads(df.T.to_json()).values()
db.sunspots.insert_many(rows)

cursor = db['sunspots'].find({})
df = pd.DataFrame(list(cursor))
print(df)

db.drop_collection('sunspots')



