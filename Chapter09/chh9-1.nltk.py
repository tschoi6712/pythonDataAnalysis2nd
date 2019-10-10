"""
문자 데이터와 소셜미디어 분석하기(1)
"""
# pip install nltk scikit-learn
# >python
# >>>import nltk
# >>>nltk.download()


## 1. Filtering out stopwords, names, and numbers(불용어, 고유명사, 숫자 걸러내기)
import nltk

# 영어 말뭉치를 불러오기
sw = set(nltk.corpus.stopwords.words('english'))
print("Stop words:", list(sw)[:7])

# 구텐베르크 말뭉치 불러오기
gb = nltk.corpus.gutenberg
print("Gutenberg files:\n", gb.fileids()[-5:])

# 파일에서 몇 개의 문장을 추출
text_sent = gb.sents("milton-paradise.txt")[:2]
print("Unfiltered:", text_sent)

# 추출된 문장에서 불용어를 걸러내기
for sent in text_sent:
    filtered = [w for w in sent if w.lower() not in sw]
    print("Filtered:\n", filtered)
    tagged = nltk.pos_tag(filtered)
    print("Tagged:\n", tagged)

    words = []
    for word in tagged:
        if word[1] != 'NNP' and word[1] != 'CD':
            words.append(word[0])

    print("Words:\n", words)



## 2. Bag of words model(단어 주머니 모델)

import sklearn as sk

# 말뭉치에서 두개의 문서를 불러오기
hamlet = gb.raw("shakespeare-hamlet.txt")
macbeth = gb.raw("shakespeare-macbeth.txt")

# 불용어를 제외하고 피처벡터를 생성
cv = sk.feature_extraction.text.CountVectorizer(stop_words='english')
print("Feature vector:\n", cv.fit_transform([hamlet, macbeth]).toarray())

# 두 문서 사이에서 피쳐(유일한 단어)를 출력
print("Features:\n", cv.get_feature_names()[:5])



## 3. Analyzing word frequencies(단어 빈도수 분석)
import nltk
import string

gb = nltk.corpus.gutenberg
words = gb.words("shakespeare-caesar.txt")

sw = set(nltk.corpus.stopwords.words('english'))
punctuation = set(string.punctuation)
filtered = [w.lower() for w in words if w.lower() not in sw and w.lower() not in punctuation]

fd = nltk.FreqDist(filtered)
print("Words", list(fd.keys())[:5])
print("Counts", list(fd.values())[:5])
print("Max", fd.max())
print("Count", fd['d'])

fd = nltk.FreqDist(nltk.bigrams(filtered))
print("Bigrams", list(fd.keys())[:5])
print("Counts", list(fd.values())[:5])
print("Bigram Max", fd.max())
print("Bigram count", fd[('let', 'vs')])



## 4. Naive Bayesian(나이브 베이즈 분류기)
import nltk
import string
import random

sw = set(nltk.corpus.stopwords.words('english'))
punctuation = set(string.punctuation)


def word_features(word):
    return {'len': len(word)}


def isStopword(word):
    return word in sw or word in punctuation


gb = nltk.corpus.gutenberg
words = gb.words("shakespeare-caesar.txt")

# 단어에 라벨 붙이기
labeled_words = ([(word.lower(), isStopword(word.lower())) for word in words])
random.seed(42)
random.shuffle(labeled_words)
print(labeled_words[:5])

# 각 단어별 길이를 측정
featuresets = [(word_features(n), word) for (n, word) in labeled_words]

# 데이터를 학습시키기
cutoff = int(.9 * len(featuresets))
train_set, test_set = featuresets[:cutoff], featuresets[cutoff:]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# 데이터의 단어가 분류되었는지 학습
print("'behold' class", classifier.classify(word_features('behold')))
print("'the' class", classifier.classify(word_features('the')))

# 분류 정확도
print("Accuracy", nltk.classify.accuracy(classifier, test_set))
print(classifier.show_most_informative_features(5))







