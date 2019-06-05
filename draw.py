import numpy as np
import random
import math
import matplotlib.pyplot as plt

import time
from joblib import Parallel, delayed
import multiprocessing


def norm(abc):
    abc = (abc - min(abc)) / (max(abc) - min(abc))
    return abc
print(10/3)

testwind = np.loadtxt('testwind0.txt')
testwind = testwind.reshape((29, 2))

xwind = norm(testwind[:,0])
ywind = testwind[:,1]

testuavv = np.loadtxt('testuavv0.txt')
testuavv = testuavv.reshape((61, 2))

xuavv = norm(testuavv[:,0])
yuavv = testuavv[:,1]
print (yuavv)


testaa = np.loadtxt('testaa0.txt')
testaa = testaa.reshape((19, 2))

xaa = norm(testaa[:,0])
yaa = testaa[:,1]

testuptime = np.loadtxt('testuptime0.txt')
testuptime = testuptime.reshape((11, 2))

xuptime = norm(testuptime[:,0])
yuptime = testuptime[:,1]

testwindalpha = np.loadtxt('testwindalpha0.txt')
testwindalpha = testwindalpha.reshape((31, 2))

xwa = norm(testwindalpha[:,0])
ywa = testwindalpha[:,1]

plt.plot(xuavv, yuavv, xwind, ywind, xwa, ywa, xaa, yaa,  xuptime, yuptime)
#plt.plot(xaa, yaa)
#plt.plot(xuavv, yuavv)
#plt.plot(xuptime, yuptime)
#plt.plot(xwa, ywa)
plt.legend(('UAV Velocity',   'Wind Velocity',  'Wind Direction','UAV Acceleration', 'Updating Time'), loc='best')
plt.xlabel('parameter')
plt.ylabel('Conflict Count')

plt.show()