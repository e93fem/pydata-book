import os
os.getcwd()

os.chdir('C:\\Users\\Fredrik\\Documents\\pydata-book')

import numpy as np

data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
arr1

data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
arr2

arr2.ndim
arr2.shape
arr1.dtype
arr2.dtype

np.zeros(10)
np.zeros((3, 6))
np.empty((2, 3, 2))
np.arange(15)

arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3], dtype=np.int32)
arr1.dtype
arr2.dtype

arr = np.array([1, 2, 3, 4, 5])
arr.dtype
float_arr = arr.astype(np.float64)
float_arr.dtype

arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
arr
arr.astype(np.int32)

numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
numeric_strings.astype(float)

int_array = np.arange(10)
calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)
int_array.astype(calibers.dtype)
empty_uint32 = np.empty(8, dtype='u4')
empty_uint32

arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr
arr * arr
arr - arr
1 / arr
arr ** 0.5

arr= np.arange(10)
arr
arr[5]
arr[5:8]
arr[5:8] = 12
arr

arr_slice = arr[5:8]
arr_slice[1] = 12345
arr

arr_slice[:] = 64
arr

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2]

arr2d[0][2]
arr2d[0, 2]

arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d
arr3d[0]
old_values = arr3d[0].copy()
arr3d[0] = 42
arr3d
arr3d[0] = old_values
arr3d
arr3d[1, 0]

arr[1:6]
arr2d
arr2d[:2]
arr2d[:2, 1:]
arr2d[1, :2]
arr2d[2, :1]
arr2d[:, :1]
arr2d[:2, 1:] = 0
arr2d

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
names
data
names == 'Bob'
data[names == 'Bob']
data[names == 'Bob', 2:]
data[names == 'Bob', 3]
names != 'Bob'
data[-(names == 'Bob')]
mask = (names == 'Bob') | (names == 'Will')
mask
data[mask]
data[data < 0] = 0
data
data[names != 'Joe'] = 7
data

arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i

arr
arr[[4, 3, 0, 6]]
arr[[-3, -5, -7]]

# more on reshape in Chapter 12
arr = np.arange(32).reshape((8, 4))
arr
arr[[1, 5, 7, 2], [0, 3, 1, 2]]
arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]
arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])]

arr = np.arange(15).reshape((3, 5))
arr
arr.T

arr = np.random.randn(6, 3)
np.dot(arr.T, arr)

arr = np.arange(16).reshape((2, 2, 4))
arr
arr.transpose((1, 0, 2))
arr
arr.swapaxes(1, 2)

arr = np.arange(10)
np.sqrt(arr)
np.exp(arr)

x = np.random.randn(8)
y = np.random.randn(8)
x
y

np.maximum(x, y) # element-wise maximum

arr = np.random.randn(7) * 5
arr
np.modf(arr)

points = np.arange(-5, 5, 0.01) # 1000 equally spaced points
xs, ys = np.meshgrid(points, points)
ys
xs

import matplotlib.pyplot as plt
z = np.sqrt(xs ** 2 + ys ** 2)
z

plt.imshow(z, cmap=plt.cm.gray)
plt.colorbar()
plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
plt.show()

xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])

result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
result

result = np.where(cond, xarr, yarr)
result

arr = np.random.randn(4, 4)
arr
np.where(arr > 0, 2, -2)
np.where(arr > 0, 2, arr) # set only positive values to 2

result = []
for i in range(n):
    if cond1[i] and cond2[i]:
        result.append(0)
    elif cond1[i]:
        result.append(1)
    elif cond2[i]:
        result.append(2)
    else:
        result.append(3)

np.where(cond1 & cond2, 0,
np.where(cond1, 1,
np.where(cond2, 2, 3)))

arr = np.random.randn(5, 4) # normally-distributed data
arr
arr.mean()
np.mean(arr)
arr.sum()
arr.mean(axis=1)
arr.mean(axis=0)
arr.sum(0)
arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr.cumsum(0)
arr.cumprod(1)

arr = np.random.randn(100)
(arr > 0).sum() # Number of positive values
bools = np.array([False, False, True, False])
bools.any()
bools.all()

arr = np.random.randn(8)
arr
arr.sort()
arr

arr = np.random.randn(5, 3)
arr
arr.sort(1)
arr

large_arr = np.random.randn(1000)
large_arr.sort()
large_arr[int(0.05 * len(large_arr))] # 5% quantile

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)
ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
np.unique(ints)
sorted(set(names))

values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])

arr = np.arange(10)
np.save('some_array', arr)
np.load('some_array.npy')

np.savez('array_archive.npz', a=arr, b=arr)
arch = np.load('array_archive.npz')
arch['b']

!more array_ex.txt
arr = np.loadtxt('array_ex.txt', delimiter=',')
arr

x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x
y
x.dot(y) # equivalently np.dot(x, y)
np.dot(x, np.ones(3))

from numpy.linalg import inv, qr
X = np.random.randn(5, 4)
mat = X.T.dot(X)
inv(mat)
mat.dot(inv(mat))
q, r = qr(mat)
r

samples = np.random.normal(size=(4, 4))
samples

from random import normalvariate
N = 1000000
%timeit samples = [normalvariate(0, 1) for _ in xrange(N)]
%timeit np.random.normal(size=N)

import random
position = 0
walk = [position]
steps = 1000
for i in xrange(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)

import matplotlib.pyplot as plt
plt.plot(walk)
plt.ylabel('Random walk')
plt.show()

nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
plt.plot(walk)
plt.ylabel('Random walk')
plt.show()

walk.min()
walk.max()
(np.abs(walk) >= 10).argmax()
