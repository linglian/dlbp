import numpy as np

root = '/home/lol/dl/dlbp/image/examples/cat/'

test = np.load(root + 'test.npy')
train = np.load(root + 'train.npy')

for (i, j) in zip(test, train):
    print '============================================================='
    print i
    print j
    print '============================================================='