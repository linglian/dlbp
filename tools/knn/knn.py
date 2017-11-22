# coding=utf-8
import numpy as np
import os
import time
import cv2


def load_all_img(path):
    import time

    subfolders = [folder for folder in os.listdir(
        path) if os.path.isdir(os.path.join(path, folder))]
    print subfolders

    for file in subfolders:
        filepath = os.path.join(path, file)
        subfolders2 = [folder for folder in os.listdir(
            filepath) if os.path.isdir(os.path.join(filepath, folder))]
        print subfolders2
        for file2 in subfolders2:
            imgArray = []
            t1 = time.time()
            filepath2 = os.path.join(filepath, file2)
            subfolders3 = [folder for folder in os.listdir(
                filepath2) if not os.path.isdir(os.path.join(filepath2, folder)) and os.path.join(filepath2, folder).endswith('.JPG')]
            print subfolders3
            for img in subfolders3:
                filepath3 = os.path.join(filepath2, img)
                print filepath3
                m = cv2.imread(filepath3, 1)
                im = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, (224, 224))
                imgArray.append([im, file2])
            print 'SpeedTime: %f' % (time.time() - t1)
            np.save(os.path.join(filepath2, 'knn.npy'), imgArray)


def load_all_beOne(path, test_ratio=0.02):
    import time

    subfolders = [folder for folder in os.listdir(
        path) if os.path.isdir(os.path.join(path, folder))]
    print subfolders

    main_imgArray = []
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
            for i in imgArray:
                main_imgArray.append(i)
        print len(main_imgArray)
    return main_imgArray[:int(len(main_imgArray) * test_ratio)], main_imgArray[int(len(main_imgArray) * test_ratio):]


def getDistances(form, to):
    return np.sqrt(np.sum(np.square(form - to)))


if __name__ == '__main__':
    import sys
    import getopt
    path = '/home/lol/dl/Image'

    opts, args = getopt.getopt(sys.argv[1:], 'f:slt')
    for op, value in opts:
        if op == '-f':
            path = value
        elif op == '-s':
            load_all_img(path)
        elif op == '-l':
            test, train = load_all_beOne(path)
            np.save(os.path.join(path, 'knn_test.npy'), test)
            np.save(os.path.join(path, 'knn_train.npy'), train)
        elif op == '-t':
            test = np.load(os.path.join(path, 'knn_test.npy'))
            train = np.load(os.path.join(path, 'knn_train.npy'))
            testNum = len(test)
            trainNum = len(train)
            print train[:]
            right = 0
            bad = 0
            now = 0
            print 'test: %d  train: %d' % (testNum, trainNum)
            for i in test:
                t1 = time.time()
                minD = [999, 'None']
                tempI = np.ravel(i[0])
                for j in train:
                    tempJ = np.ravel(j[0])
                    dist = getDistances(tempI, tempJ)
                    if dist < minD[0]:
                        minD[0] = dist
                        minD[1] = j[1]
                if minD[1] == i[1]:
                    right += 1
                else:
                    bad += 1
                now += 1
                print 'right: %d bad: %d now: %d/%d Time: %f s' % (right, bad, now, testNum, (time.time() - t1))
