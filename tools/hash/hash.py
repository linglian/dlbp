# coding=utf-8
import numpy as np
import os
import time
import cv2
import logging
import sys
sys.path.append('/home/lol/anaconda2/lib/python2.7/site-packages')
import imagehash as ih
from PIL import Image

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='test.log',
                filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def getHash(img):
    im = Image.fromarray(img)
    return ih.average_hash(im, 8)

def load_all_beOne(path, test_ratio=0.02):
    import time
    import random
    subfolders = [folder for folder in os.listdir(
        path) if os.path.isdir(os.path.join(path, folder))]
    print subfolders
    tt = time.time()
    main_imgArray = []
    print 'Start Merge Npy'
    for file in subfolders:
        filepath = os.path.join(path, file)
        subfolders2 = [folder for folder in os.listdir(
            filepath) if os.path.isdir(os.path.join(filepath, folder))]
        print subfolders2
        imgArray = []
        for file2 in subfolders2:
            t1 = time.time()
            filepath2 = os.path.join(filepath, file2)
            imgArray = np.load(os.path.join(filepath2, 'knn.npy'))
            if len(imgArray) == 0:
                logging.error('Bad Npy: %s' % os.path.join(filepath2, 'knn.npy'))
            for i in imgArray:
                main_imgArray.append([getHash(i[0]), i[1]])
        print 'End Merge Npy: %d Speed Time: %f s' % (len(main_imgArray), (time.time() - tt))
    random.shuffle(main_imgArray)
    return main_imgArray[:int(len(main_imgArray) * test_ratio)], main_imgArray[int(len(main_imgArray) * test_ratio):]

def getDistances(f, t):
    return f[0] - t[0]

def getMinOfNum(a, K):
    a = np.array(a)
    return np.argpartition(a,-K)[-K:]

if __name__ == '__main__':
    import sys
    import getopt
    from collections import Counter
    path = '/home/lol/dl/Image'
    test, train = load_all_beOne(path, test_ratio=0.02)
    testNum = len(test)
    trainNum = len(train)
    right = 0
    bad = 0
    now = 0
    print 'Start Test (Test: %d Train: %d)' % (len(test), len(train))
    m_time = time.time()
    for i in test:
        t1 = time.time()
        minD = []
        tempI = np.ravel(i[0])
        for j in train:
            tempJ = np.ravel(j[0])
            dist = getDistances(tempI, tempJ)
            print dist
            minD.append([dist, j[1]])
        label = [x[1] for x in minD]
        label = np.array(label)
        tempArray1 = np.array(getMinOfNum([x[0] for x in minD], 5))
        tempArray2 = label[tempArray1]
        la = Counter(tempArray2).most_common(1)[0][0]
        if la == i[1]:
            right += 1
        else:
            bad += 1
            logging.warn('bad: %s != %s' % (i[1], la))
        now += 1
        logging.info('right: %d bad: %d now: %d/%d Time: %f s' % (right, bad, now, testNum, (time.time() - t1)))
    
    print 'End Test Speed Time: %f s' % (time.time() - m_time)
