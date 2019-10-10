"""
데이터의 검색, 처리, 저장(2)
"""


## 6. JSON

# (1) Using REST and JSON
import json

json_str = '{"country":"Netherlands",' \
           '"dma_code":"0",' \
           '"timezone":"Europe\/Amsterdam",' \
           '"area_code":"0",' \
           '"ip":"46.19.37.108",' \
           '"asn":"AS196752",' \
           '"continent_code":"EU",' \
           '"isp":"Tilaa V.O.F.",' \
           '"longitude":5.75,' \
           '"latitude":52.5,' \
           '"country_code":"NL",' \
           '"country_code3":"NLD"' \
           '}'

# JSON 데이터 읽고 접근하기
data = json.loads(json_str)
print("Country", data["country"])

# JSON 데이터 수정하고 쓰기
data["country"] = "Brazil"
print(json.dumps(data))

# (2) 판다스와 JSON 읽기와 쓰기
import pandas as pd

# 판다스의 시리즈 만들기
data = pd.read_json(json_str, typ='series')
print("Series\n", data)

# 판다스 시리즈를 JSON 문자열로 변환
data["country"] = "Brazil"
print("New Series\n", data.to_json())

## 7. Parsing RSS and Atom Feeds(RSS와 Atom 피드 파싱)
# pip install feedparser

import feedparser as fp

rss = fp.parse("http://www.packtpub.com/rss.xml")
print("# Entries", len(rss.entries))

# 목록의 제목과 요약을 출력
for i, entry in enumerate(rss.entries):
    if "Python" in entry.summary:
        print(i, entry.title)
        print(entry.summary)

## 8. Parsing HTML with Beautiful Soup
# pip install beautifulsoup4 lxml

from bs4 import BeautifulSoup
import re

path = 'D:\\1.Workspace\\1.Python\\part2.data_analysis\\pythonDataAnalysis2nd\\'
soup = BeautifulSoup(open(path + 'ch5-2.loremIpsum.html'), "lxml")

print("First div: \n", soup.div)                     # 첫번째 div 태그 엘리먼트에 접근
print("First div class: ", soup.div['class'])        # 첫번째 div 태그의 속성에 접근
print("First dfn text: ", soup.dl.dt.dfn.text)       # .으로 원하는 하위 엘리먼트에 접근

# HTML의 하이퍼링크를 찾고 모두 출력
for link in soup.find_all('a'):
    print("Link text: ", link.string, "URL: ", link.get('href'))

# find_all() 을 생략하고 모든 div 내용을 출력
for i, div in enumerate(soup('div')):
    print(i, div.contents)

# id=official 인 div 엘리먼트를 선택
official_div = soup.find_all("div", id="official")
print("Official Version: ", official_div[0].contents[2].strip())

print("# elements with class: ", len(soup.find_all(class_=True)))

# class=tile 인 div 엘리먼트를 선택
tile_class = soup.find_all("div", class_="tile")
print("# Tile classes: ", len(tile_class))

print("# Divs with class containing tile: ", len(soup.find_all("div", class_=re.compile("tile"))))

# select() 메서드
print("Using CSS selector: \n", soup.select('div.notile'))
print("Selecting ordered list list items: \n", soup.select("ol > li")[:2])
print("Second list item in ordered list: ", soup.select("ol > li:nth-of-type(2)"))

print("Searching for text string: ", soup.find_all(text=re.compile("2014")))
