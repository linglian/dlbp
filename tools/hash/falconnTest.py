import sys
sys.path.insert(0, '/home/lol/anaconda2/lib/python2.7/site-packages')
import falconn
import numpy as np
import os
import time


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


path = '/home/lol/dl/Image'
test_name = 'feature'
mxnetpath = '/home/lol/dl/mxnet/python'
times = 1
resetTest = False
test_ratio = 0.02
reportTime = 10
is_big_key = False
is_log = False
ks = {}
k = 1


def resetRandom():
    import random
    test = np.load(os.path.join(path, test_name + '_test.npy'))
    train = np.load(os.path.join(path, test_name + '_train.npy'))
    testNum = len(test)
    trainNum = len(train)
    num = testNum + trainNum
    tempList = []
    # print 'Start Random: %d + %d = %d' % (testNum, trainNum, num)
    for i in test:
        tempList.append(i)
    for i in train:
        tempList.append(i)
    random.shuffle(tempList)
    # print 'End Random: %d + %d = %d' % (testNum, trainNum, num)
    np.save(os.path.join(path, test_name + '_test.npy'),
            tempList[:int(num * test_ratio)])
    np.save(os.path.join(path, test_name + '_train.npy'),
            tempList[int(num * test_ratio):])


def runTest():
    m_bad = 0
    m_right = 0
    m_num = 0
    for main_times in range(0, 10):
        if resetTest:
            resetRandom()
        test = np.load(os.path.join(path, test_name + '_test.npy'))
        train = np.load(os.path.join(path, test_name + '_train.npy'))
        testNum = len(test)
        trainNum = len(train)
        p = falconn.get_default_parameters(trainNum, 2048)
        t = falconn.LSHIndex(p)
        dataset = [np.ravel(x[0]).astype(np.float32) for x in train]
        dataset = np.array(dataset)
        t.setup(dataset)
        q = t.construct_query_object()

        for i in test:
            t_num = 0
            t1 = time.time()
            # print i[1], i[2]
            tList = train[q.find_k_nearest_neighbors(i[0], k)]
            is_true = False
            for l in tList:
                if is_big_key:
                    if ks[l[1]] == ks[i[1]]:
                        is_true = True
                        break
                else:
                    if l[1] == i[1]:
                        is_true = True
                        break
            if is_true:
                m_right += 1
            else:
                m_bad += 1
                if is_log:
                    if is_big_key:
                        logging.error('###### Bad %s(%s: %s) with %s' (
                            ks[i[1]], i[1], i[2], tList))
                    else:
                        logging.error(
                            '###### Bad %s: %s with %s' (i[1], i[2], tList))
            m_num += 1
            t_num += 1
            if m_num % reportTime == 1:
                logging.info('Last accuracy: %.2f %%' %
                             (m_right / float(m_num) * 100.0))
                logging.info('Last loss: %.2f %%' %
                             (m_bad / float(m_num) * 100.0))
                t_min = min(t_num, reportTime)
                logging.info('right: %d bad: %d now: %d/%d Time: %.2fs/%diter' %
                             (m_right, m_bad, m_num, testNum * times, (time.time() - t1), t_min))
    logging.info('Last accuracy: %.2f %% (%d/%d)' % ((m_right / float(m_num) * 100.0), m_right, m_num))
    logging.info('Last loss: %.2f %% (%d/%d)' % ((m_bad / float(m_num) * 100.0), m_bad, m_num))
    logging.info('End Run Test')


runTest()

# if __name__ == '__main__':
#     import sys
#     import getopt
#     from collections import Counter
#     import random

#     opts, args = getopt.getopt(sys.argv[1:], 'f:sltzr:ai:mk:gx:v:hb', ['time=', 'dist=', 'report=', 'hash', 'size', 'log', 'round='])
#     for op, value in opts:
#         if op == '-f':
#             path = value
#         elif op == '-h':
#             resetTest = True
#         elif op == '-v':
#             test_name = value
#         elif op == '-g':
#             loadFeature()
#         elif op == '--log':
#             is_log = True
#         elif op == '-b':
#             is_big_key = True
#             subfolders = [folder for folder in os.listdir(
#                 path) if os.path.isdir(os.path.join(path, folder))]
#             for file in subfolders:
#                 print 'Start %s' % file
#                 path2 = os.path.join(path, file)
#                 subfolders2 = [folder for folder in os.listdir(
#                     path2) if os.path.isdir(os.path.join(path2, folder))]
#                 for file2 in subfolders2:
#                     if ks.has_key(file2):
#                         print '######### Error Has Same: %s(%s) %s' % (file, file2, ks[file2])
#                     ks[file2] = file
#                 print 'End %s' % file
#         elif op == '--hash':
#             loadHash()
#         elif op == '-x':
#             mxnetpath = value
#             sys.path.insert(0, mxnetpath)
#         elif op == '-k':
#             k = int(value)
#         elif op == '--time':
#             times = int(value)
#         elif op == '--report':
#             reportTime = int(value)
#         elif op == '-r':
#             test_ratio = float(value)
#         elif op == '--size':
#             test = np.load(os.path.join(path, test_name + '_test.npy'))
#             train = np.load(os.path.join(path, test_name + '_train.npy'))
#             print 'Size: %d' % (len(test) + len(train))
#         elif op == '-t':
#             runTest()