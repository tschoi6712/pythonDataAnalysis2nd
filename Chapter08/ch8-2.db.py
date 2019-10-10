"""
데이터베이스로 작업하기(2)
"""

## 3. SQLAlchemy - 데이터베이스 테이블을 매핑하는 디자인 패턴에 기반한 객체관계매핑

#(1) SQLAlchemy 설치하고 구성하기
from sqlalchemy import Column, ForeignKey, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy import UniqueConstraint


Base = declarative_base()
class Station(Base):
    __tablename__ = 'station'
    id = Column(Integer, primary_key=True)
    name = Column(String(14), nullable=False, unique=True)

    def __repr__(self):
        return "Id=%d name=%s" %(self.id, self.name)

class Sensor(Base):
    __tablename__ = 'sensor'
    id = Column(Integer, primary_key=True)
    last = Column(Integer)
    multiplier = Column(Float)
    station_id = Column(Integer, ForeignKey('station.id'))
    station = relationship(Station)

    def __repr__(self):
        return "Id=%d last=%d multiplier=%.1f station_id=%d"
        # %(self.id, self.last, self.multiplier, self.station_id)

if __name__ == "__main__":
    print("This script is used by code further down in this notebook.")



#(2) SQLAlchemy 로 데이터베이스 추가하기
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
#from alchemy_entities import Base, Sensor, Station


def populate(engine):
    Base.metadata.bind = engine

    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    de_bilt = Station(name='De Bilt')
    session.add(de_bilt)
    session.add(Station(name='Utrecht'))
    session.commit()
    print("Station", de_bilt)

    temp_sensor = Sensor(last=20, multiplier=.1, station=de_bilt)
    session.add(temp_sensor)
    session.commit()
    print("Sensor", temp_sensor)

if __name__ == "__main__":
    print("This script is used by code further down in this notebook")



#(3) SQLAlchemy 로 데이터베이스 쿼리하기
#from alchemy_entities import Base, Sensor, Station
#from populate_db import populate
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from pandas.io.sql import read_sql


engine = create_engine('sqlite:///demo.db')
Base.metadata.create_all(engine)
populate(engine)
Base.metadata.bind = engine
DBSession = sessionmaker()
DBSession.bind = engine
session = DBSession()

station = session.query(Station).first()

print("Query 1", session.query(Station).all())
print("Query 2", session.query(Sensor).all())
print("Query 3", session.query(Sensor).filter(Sensor.station == station).one())
print(read_sql("SELECT * FROM station", engine.raw_connection()))

try:
    os.remove('demo.db')
    print("Deleted demo.db")
except OSError:
    pass


























