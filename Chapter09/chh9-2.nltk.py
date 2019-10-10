"""
문자 데이터와 소셜미디어 분석하기(2)
"""


## 5. Sentiment Analysis(감성분석) - 영화 리뷰 분류(긍정적과 부정적)
import random
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy
import string


# 영화 리뷰 문서 라벨링
labeled_docs = [(list(movie_reviews.words(fid)), cat)
        for cat in movie_reviews.categories()
        for fid in movie_reviews.fileids(cat)]
random.seed(42)
random.shuffle(labeled_docs)

review_words = movie_reviews.words()
print("# Review Words", len(review_words))

sw = set(stopwords.words('english'))
punctuation = set(string.punctuation)

def isStopWord(word):
    return word in sw or word in punctuation

filtered = [w.lower() for w in review_words if not isStopWord(w.lower())]
print("# After filter", len(filtered))

# 가장 빈도수가 높은 상위 5%의 단어
words = FreqDist(filtered)
N = int(.05 * len(words.keys()))
word_features = list(words.keys())[:N]

# 단어 갯수를 측정 기준으로 삼는 함수
def doc_features(doc):
    doc_words = FreqDist(w for w in doc if not isStopWord(w))
    features = {}
    for word in word_features:
        features['count (%s)' % word] = (doc_words.get(word, 0))
    return features

featuresets = [(doc_features(d), c) for (d,c) in labeled_docs]
train_set, test_set = featuresets[200:], featuresets[:200]
classifier = NaiveBayesClassifier.train(train_set)
print("Accuracy", accuracy(classifier, test_set))

print(classifier.show_most_informative_features())



## 6. Creating Word Clouds(워드 클라우드 만들기)
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.corpus import names
from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
import pandas as pd
import numpy as np
import string

sw = set(stopwords.words('english'))
punctuation = set(string.punctuation)
all_names = set([name.lower() for name in names.words()])

def isStopWord(word):
    return (word in sw or word in punctuation) or not word.isalpha() or word in all_names

review_words = movie_reviews.words()
filtered = [w.lower() for w in review_words if not isStopWord(w.lower())]

words = FreqDist(filtered)

texts = []

for fid in movie_reviews.fileids():
    texts.append(" ".join([w.lower() for w in movie_reviews.words(fid)
                           if not isStopWord(w.lower()) and words[w.lower()] > 1]))

vectorizer = TfidfVectorizer(stop_words='english')
matrix = vectorizer.fit_transform(texts)
sums = np.array(matrix.sum(axis=0)).ravel()

ranks = []

for word, val in zip(vectorizer.get_feature_names(), sums):
    ranks.append((word, val))

df = pd.DataFrame(ranks, columns=["term", "tfidf"])
df = df.sort_values(['tfidf'])
print(df.head())

N = int(.01 * len(df))
df = df.tail(N)

for term, tfidf in zip(df["term"].values, df["tfidf"].values):
    print(term, ":", tfidf)



## 7. Social Network Analysis(소셜미디어 분석)
#pip install networkx

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


print([s for s in dir(nx) if s.endswith('graph')])

G = nx.davis_southern_women_graph()
plt.hist(list(dict(nx.degree(G)).values()))
plt.show()

plt.figure(figsize=(8,8))
pos = nx.spring_layout(G)
nx.draw(G, node_size=10)
nx.draw_networkx_labels(G, pos)
plt.show()



