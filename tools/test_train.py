#coding=utf-8
import numpy as np
import os
import math
import multiprocessing
import sys

import logging

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

def EuclideanDistance(form, to):
    sum = 0
    for (i, j) in zip(form, to):
        sum += (i - j) * (i - j)
    return math.sqrt(sum)

def run():
    import time
    t = time.time()
    filePath = '/home/lol/dl/dlbp/image'
    print "Start load npy"
    test = np.load(os.path.join(filePath, 'test.npy'))
    train = np.load(os.path.join(filePath, 'train.npy'))
    print "End load npy"
    testNum = 0
    for i in test:
        for j in i:
            testNum += 1
    trainNum = 0
    for i in train:
        for j in i:
            trainNum += 1
    logging.info('TestNumber: %d TrainNumber: %d' % (testNum, trainNum))
    testNow = 0
    good = 0
    bad = 0
    for i in test:
        for j in i:
            t1 = time.time()
            logging.info('Start: %s %s' % (j[1], j[2]))
            minD = [999, '', ''];
            trainNow = 0
            for k in train:
                for l in k:
                    temp = EuclideanDistance(j[0], l[0])
                    if minD[0] > temp:
                        minD[0] = temp
                        minD[1] = l[1]
                        minD[2] = l[2]
                    trainNow += 1
                    #if trainNow % 500 == 0:
                    #    logging.info('Finish Train %d/%d' % (trainNow, trainNum))
            logging.info('End: %f %s %s (Speed time: %f)' % (minD[0], minD[1], minD[2], time.time() - t1))
            testNow += 1
            if j[1] == minD[1]:
                good += 1
            else:
                bad += 1
                logging.warning('Bad is Coming: %s %s != %s %s' %(j[1], [2], minD[1], minD[2]))
            logging.info('Good/Bad %d/%d' % (good, bad))
            logging.info('Finish Test %d/%d' % (testNow, testNum))
    logging.info('Finish All, Speed time: %f' % (time.time() - t))
if __name__=='__main__':
    jobs = []
    print '########### Start Process'
    p = multiprocessing.Process(target=run)
    jobs.append(p)
    p.start()
    for p in jobs:
        print "End Process"
        p.join()