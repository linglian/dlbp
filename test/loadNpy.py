import numpy as np
import os
import sys

root = '/home/lol/dl/dlbp/image/examples'

npy_paths = [os.path.join(root, npy) for npy in os.listdir(
    root) if os.path.isdir(os.path.join(root, npy))]

print npy_paths

i = 0

def checkFold(name):
    if os.path.exists(name):
        os.remove(name)
    os.mknod(name)

checkFold(root + '/test.lst')

out = open(root + '/test.lst', 'w')

checkFold(root + '/train.lst')

out2 = open(root + '/train.lst', 'w')

for npy_path in npy_paths:
    test = np.load(os.path.join(npy_path, 'test.npy'))
    train = np.load(os.path.join(npy_path, 'test.npy'))
    for (te, tr) in zip(test, train):
        out.write('%d\t%d\t%s\n' % (i, te[0].argmax(), os.path.join(npy_path, te[2])))
        out2.write('%d\t%d\t%s\n' % (i, tr[0].argmax(), os.path.join(npy_path, tr[2])))
        i += 1

out.close()
out2.close()