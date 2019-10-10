import fort_sum
import numpy as np

rain = np.load('ch10.rain.npy')
fort_sum.sumarray(rain, len(rain))
rain = .1 * rain
rain[rain < 0] = .025
print("Numpy", rain.sum())
