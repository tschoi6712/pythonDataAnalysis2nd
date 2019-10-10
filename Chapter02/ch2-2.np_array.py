"""
Numpy 배열 다루기(2)
"""

## 7. creating array views and copies(배열 뷰 생성하고 복사하기)
import scipy.misc
import matplotlib.pyplot as plt

face = scipy.misc.face()                        # 얼굴 이미지 불러오기
acopy = face.copy()                             # 배열의 복사본 만들기
aview = face.view()                             # 배열의 뷰 만들기
aview.flat = 0

plt.subplot(221)
plt.imshow(face)
plt.subplot(222)
plt.imshow(acopy)
plt.subplot(223)
plt.imshow(aview)
plt.show()



## 8. fancy indexing - 대각선 값을 0으로 한 고급 인덱싱
import scipy.misc
import matplotlib.pyplot as plt

face = scipy.misc.face()                        #
xmax = face.shape[0]
ymax = face.shape[1]

face = face[:min(xmax, ymax), :min(xmax, ymax)]
xmax = face.shape[0]
ymax = face.shape[1]

face[range(xmax), range(ymax)] = 0              # x와 y 값의 서로 다른 범위(대각선 값 모두 0)
face[range(xmax-1, -1, -1), range(ymax)] = 0    # x와 y 값의 서로 다른 범위(반대쪽 대각선 값 모두 0)

plt.imshow(face)
plt.show()



## 9. indexing with list of locations(위치 데이터로 인덱싱 하기)
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np

face = scipy.misc.face()
xmax = face.shape[0]
ymax = face.shape[1]

def shuffle_indices(size):
    """
    위치 기반의 리스트로 넘파이 배열화 인덱스 작업
    """
    arr = np.arange(size)
    np.random.shuffle(arr)
    return arr

xindices = shuffle_indices(xmax)
np.testing.assert_equal(len(xindices), xmax)

yindices = shuffle_indices(ymax)
np.testing.assert_equal(len(yindices), ymax)

plt.imshow(face[np.ix_(xindices, yindices)])    # np.ix_() 행과 열에 대응하는 모양의 값을 선택 - 메시 데이터
plt.show()



## 10. indexing arrays with booleans(논리향 방식으로 인덱싱하기)
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np

face = scipy.misc.face()
xmax = face.shape[0]
ymax = face.shape[1]
face = face[:min(xmax, ymax), :min(xmax, ymax)]

def get_indices(size):
    """
    대각선의 값들 중 4의 배수를 선택
    """
    arr = np.arange(size)
    return arr % 4 == 0

face1 = face.copy()
xindices = get_indices(face.shape[0])
yindices = get_indices(face.shape[1])
face1[xindices, yindices] = 0

plt.subplot(211)
plt.imshow(face1)

face2 = face.copy()                             # 최대값의 1/4부터 3/4까지에 해당하는 배열의 값 = 0
face2[(face > face.max()/4) & (face < 3 * face.max()/4)] = 0

plt.subplot(212)
plt.imshow(face2)
plt.show()



## 11. broadcasting arrays(브로드캐스팅) - 데이터 형이 확장 + 데이터 형이 변환
import scipy.io.wavfile as sw
import matplotlib.pyplot as plt
import urllib.request
import numpy as np

"""
http 요청과 응답
urllib.request.Request(url, data=None, headers={}, origin_req_host=None, unverifiable=False, method=None)
urllib.request.urlopen(url, data=None, [timeout, ]*, cafile=None, capath=None, cadefault=False, context=None)
"""
request = urllib.request.Request('http://www.thesoundarchive.com/austinpowers/smashingbaby.wav')
response = urllib.request.urlopen(request)
print(response.info())


# SciPy의 wavfile 패키지로 오디오 파일을 불러와 .wav파일을 생성
WAV_FILE = 'ch2-2.data.smashingbaby.wav'
filehandle = open(WAV_FILE, 'wb')
filehandle.write(response.read())
filehandle.close()
sample_rate, data = sw.read(WAV_FILE)               # 배열 데이터와 오디오 샘플 속도를 전달 - read()
print("Data type:", data.dtype, "Shape:", data.shape)

# 기존 .wav 데이터 그래프화 하기
plt.subplot(2, 1, 1)
plt.title("Original")
plt.plot(data)

# 새로운 배열 생성하기
newdata = data * 0.2
newdata = newdata.astype(np.uint8)
print("Data type: ", newdata.dtype, "Shape: ", newdata.shape)

# .wav 파일 만들기
sw.write("ch2-2.data.quiet.wav", sample_rate, newdata)

# 새로운 .wav 데이터 그래프화 하기
plt.subplot(2, 1, 2)
plt.title("Quiet")
plt.plot(newdata)

plt.show()


