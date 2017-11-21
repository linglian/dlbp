import numpy as np
import os
import sys

root = '/home/lol/dl/dlbp/image/examples'

npy_paths = [os.path.join(root, npy) for npy in os.listdir(
    root) if os.path.isdir(os.path.join(root, npy))]

print npy_paths

i = 0
def checkFile(name):
    if os.path.exists(name):
        os.remove(name)
    os.mknod(name)

def checkFold(name):
    if not os.path.exists(name):
        os.mkdir(name)

checkFold(root + '/../lst')

checkFile(root + '/../lst/test.lst')

out = open(root + '/../lst/test.lst', 'w')

checkFile(root + '/../lst/train.lst')

out2 = open(root + '/../lst/train.lst', 'w')

for npy_path in npy_paths:
    test = np.load(os.path.join(npy_path, 'test.npy'))
    train = np.load(os.path.join(npy_path, 'test.npy'))
    for (te, tr) in zip(test, train):
        out.write('%d\t%d\t%s\n' % (i, te[0].argmax(), os.path.join(te[1], te[2])))
        out2.write('%d\t%d\t%s\n' % (i, tr[0].argmax(), os.path.join(tr[1], tr[2])))
        i += 1

out.close()
out2.close()