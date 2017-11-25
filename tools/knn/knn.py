#coding=utf-8
import numpy as np
import os
import time
import cv2
import logging
import shutil
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


path = '/home/lol/dl/Image'
not_double = True
test_ratio = 0.02
tilesPerImage = 1
k = 1
times = 1
mxnetpath = '/home/lol/dl/mxnet/python'
test_name = 'knn'
sys.path.insert(0, mxnetpath)
resetTest = False
distType = 1
reportTime = 500
is_big_key = False
ks = {}
is_log = False
num_round = 0

def checkFold(name):
    if not os.path.exists(name):
        os.mkdir(name)


def removeDir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)


def removeFile(name):
    if os.path.exists(name):
        os.remove(name)

def getHash(img):
    im = Image.fromarray(img)
    return ih.average_hash(im, 8)

def splits_resamples(facescrub_root, tilesPerImage=360):
    #import sklearn
    thresholdGLOABL = 0.42
    from PIL import Image
    import random
    import math

    fold = facescrub_root
    print fold

    subfolders = [folder for folder in os.listdir(
        facescrub_root) if os.path.isdir(os.path.join(facescrub_root, folder))]
    print subfolders

    dict = {}
    for subfolder in subfolders:
        removeFile(os.path.join(facescrub_root, subfolder, 'test.npy'))
        removeFile(os.path.join(facescrub_root, subfolder, 'train.npy'))
        imgsfiles = [os.path.join(facescrub_root, subfolder, img)
                     for img in os.listdir(os.path.join(facescrub_root, subfolder)) if img.endswith('.JPG')]
        for img in imgsfiles:
            dict[img] = subfolder
    print dict

    def im_crotate_image_square(im, deg):
        im2 = im.rotate(deg, expand=1)
        im = im.rotate(deg, expand=0)

        width, height = im.size
        if width == height:
            im = im.crop((0, 0, w, int(h * 0.9)))
            width, height = im.size

        rads = math.radians(deg)
        new_width = width - (im2.size[0] - width)

        left = top = int((width - new_width) / 2)
        right = bottom = int((width + new_width) / 2)

        return im.crop((left, top, right, bottom))


    dx = dy = 224
    fold_idx = 1
    rotateAction = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
                    Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

    rotate45degree = [45, 135, 270]
    subfolders = [folder for folder in os.listdir(
        fold) if os.path.isdir(os.path.join(fold, folder))]
    print 'files: %s' % subfolders

    for subfolder in subfolders:
        imgsfiles = [os.path.join(fold, subfolder, img)
                     for img in os.listdir(os.path.join(fold, subfolder)) if img.endswith('.JPG')]
        print 'Start Directory: %s' % subfolder
        for imgfile in imgsfiles:
            print 'Start Image: %s' % imgfile
            im = Image.open(imgfile)
            w, h = im.size
            im = im.crop((0, 0, w, int(h * 0.9)))
            #dx = 224
            for i in range(1, tilesPerImage + 1):
                newname = imgfile.replace('.', '_{:03d}.'.format(i))
                # print newname
                w, h = im.size
                if w < 224:
                        im = cv2.resize(im, (224, h))
                w, h = im.size
                if h < 224:
                        im = cv2.resize(im, (w, 224))
                w, h = im.size

                # print("Cropping",w,h)
                if i < 100 and w > 300:
                    dx = 224
                if 100 < i < 200 and w > 500:
                    dx = 320
                if 200 < i < 300 and w > 800:
                    dx = 640
                if i < 100 and h > 300:
                    dy = 224
                if 100 < i < 200 and h > 500:
                    dy = 320
                if 200 < i < 300 and h > 800:
                    dy = 640
                x = random.randint(0, w - dx - 1)
                y = random.randint(0, h - dy - 1)
                #print("Cropping {}: {},{} -> {},{}".format(file, x,y, x+dx, y+dy))
                im_cropped = im.crop((x, y, x + dx + 5, y + dy + 5))
                if i % 2 == 0:  # roate 180,90
                    im_cropped = im_cropped.transpose(
                        random.choice(rotateAction))
                if i % 2 == 0 and i > 300:
                    roate_drgree = random.choice(rotate45degree)
                    im_cropped = im_crotate_image_square(
                        im_cropped, roate_drgree)
                if w != 0 and h != 0:
                    im_cropped.save(newname)
            # don't remove startImg
            # os.remove(imgfile)
    return fold

def load_all_img(path, not_double=True):
    import time

    subfolders = [folder for folder in os.listdir(
        path) if os.path.isdir(os.path.join(path, folder))]
    print subfolders

    m_num = 0
    for file in subfolders:
        filepath = os.path.join(path, file)
        subfolders2 = [folder for folder in os.listdir(
            filepath) if os.path.isdir(os.path.join(filepath, folder))]
        print subfolders2
        n = 0
        for file2 in subfolders2:
            n += 1
            imgArray = []
            t1 = time.time()
            filepath2 = os.path.join(filepath, file2)
            if not_double and os.path.exists(os.path.join(filepath2, 'knn.npy')):
                if len(np.load(os.path.join(filepath2, 'knn.npy'))) != 0:
                    continue
            subfolders3 = [folder for folder in os.listdir(
                filepath2) if not os.path.isdir(os.path.join(filepath2, folder)) and os.path.join(filepath2, folder).endswith('.JPG')]
            print subfolders3
            for img in subfolders3:
                filepath3 = os.path.join(filepath2, img)
                print filepath3
                m = cv2.imread(filepath3, 1)
                if m is not None:
                    im = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
                    im = cv2.resize(m, (224, 224))
                    imgArray.append([im, file2, img])
                else:
                    logging.error('Bad Image: %s' % filepath3)
            print 'SpeedTime: %f' % (time.time() - t1)
            np.save(os.path.join(filepath2, 'knn.npy'), imgArray)
        print '%s has %d' % (file, n)
        m_num += n
    print 'Sum: %d' % m_num


def load_all_beOne(path, test_ratio=0.02):
    import time
    import random
    subfolders = [folder for folder in os.listdir(
        path) if os.path.isdir(os.path.join(path, folder))]
    print subfolders
    tt = time.time()
    main_imgArray = []
    per = 0
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
            print 'Load Knn.npy: %s' % (os.path.join(filepath2, 'knn.npy'))
            if len(imgArray) == 0:
                logging.error('Bad Npy: %s' % os.path.join(filepath2, 'knn.npy'))
            for i in imgArray:
                main_imgArray.append(i)
        print 'End Merge Npy: %d %f s' % (len(main_imgArray), (time.time() - tt))
    random.shuffle(main_imgArray)
    return main_imgArray[:int(len(main_imgArray) * test_ratio)], main_imgArray[int(len(main_imgArray) * test_ratio):]

def getDistances(f, t, type=1):
    if type == 1:
        return getDistOfL2(f, t)
    elif type == 2:
        return getDistOfHash(f, t)
    elif type == 3:
        return getDistOfSquare(f, t)
    elif type == 4:
        return 1.0 - getDistOfCos(f, t)

def getDistOfL2(form, to):
    return cv2.norm(form, to, normType=cv2.NORM_L2)

def getDistOfSquare(form, to):
    return np.sqrt(np.sum(np.square(form - to)))

def getDistOfHash(f, t):
    return f[0].__sub__(t[0])

def getDistOfCos(f, t):
    up = np.sum(np.multiply(f, t))
    ff = np.sqrt(np.sum(np.multiply(f, f)))
    tt = np.sqrt(np.sum(np.multiply(t, t)))
    down = ff * tt
    return  up / down

def getMinOfNum(a, K):
    a = np.array(a)
    return sorted(a, key=lambda a: a[0])[0:K]

def removeAllSplits(path):
    imgList = [img for img in os.listdir(path) if img.endswith('.JPG') and img.find('_') > 0]
    print 'del Img: %s' % imgList
    for i in imgList:
        removeFile(os.path.join(path, i))


def getImage(img):
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img

def getFeatures(img, f_mod):
    img = getImage(img)
    f = f_mod.predict(img)
    f = np.ravel(f)
    return f

def init(GPUid=0):
    import mxnet as mx
    prefix = "full-resnet-152"
    model = mx.model.FeedForward.load(
        prefix, num_round, ctx=mx.gpu(GPUid), numpy_batch_size=1)
    internals = model.symbol.get_internals()
    fea_symbol = internals["pool1_output"]
    feature_extractor = mx.model.FeedForward(ctx=mx.gpu(GPUid), symbol=fea_symbol, numpy_batch_size=1,
                                             arg_params=model.arg_params, aux_params=model.aux_params, allow_extra_params=True)

    return feature_extractor

def removeAllSpliteOfPath():
    subfolders = [folder for folder in os.listdir(
        path) if os.path.isdir(os.path.join(path, folder))]
    print subfolders
    for file in subfolders:
        path2 = os.path.join(path, file)
        subfolders2 = [folder for folder in os.listdir(
            path2) if os.path.isdir(os.path.join(path2, folder))]
        print subfolders2
        for file2 in subfolders2:
            t_time = time.time()
            print 'Start ImageDir: %s ' % os.path.join(path2, file2)
            removeAllSplits(os.path.join(path2, file2))
            print 'End ImageDir: %s Speed Time: %f' % (os.path.join(path2, file2), (time.time() - t_time))

def loadFeature():
    test = np.load(os.path.join(path, 'knn_test.npy'))
    train = np.load(os.path.join(path, 'knn_train.npy'))
    testNum = len(test)
    trainNum = len(train)
    m_t = time.time()
    print 'Start Feature: Test: %d Train: %d' % (testNum, trainNum)
    mod = init()
    testList = []
    n = 0
    t_time = time.time()
    for i in test:
        testList.append([getFeatures(i[0], mod), i[1], i[2]])
        n += 1
        if n % 500 == 0:
            print 'Finish %d/%d  SpeedTime: %f s' % (n, testNum, (time.time() - t_time))
            t_time = time.time()
    trainList = []
    n = 0
    t_time = time.time()
    for i in train:
        trainList.append([getFeatures(i[0], mod), i[1], i[2]])
        n += 1
        if n % 500 == 0:
            print 'Finish %d/%d  SpeedTime: %f s' % (n, trainNum, (time.time() - t_time))
            t_time = time.time()
    np.save(os.path.join(path, 'feature_test.npy'), testList)
    np.save(os.path.join(path, 'feature_train.npy'), trainList)
    print 'End Feature: Speed Time %f' % (time.time() - m_t)


def loadHash():
    test = np.load(os.path.join(path, 'knn_test.npy'))
    train = np.load(os.path.join(path, 'knn_train.npy'))
    testNum = len(test)
    trainNum = len(train)
    m_t = time.time()
    print 'Start Hash: Test: %d Train: %d' % (testNum, trainNum)
    testList = []
    n = 0
    t_time = time.time()
    for i in test:
        testList.append([getHash(i[0]), i[1], i[2]])
        n += 1
        if n % 500 == 0:
            print 'Finish %d/%d  SpeedTime: %f s' % (n, testNum, (time.time() - t_time))
            t_time = time.time()
    trainList = []
    n = 0
    t_time = time.time()
    for i in train:
        trainList.append([getHash(i[0]), i[1], i[2]])
        n += 1
        if n % 500 == 0:
            print 'Finish %d/%d  SpeedTime: %f s' % (n, trainNum, (time.time() - t_time))
            t_time = time.time()
    np.save(os.path.join(path, 'hash_test.npy'), testList)
    np.save(os.path.join(path, 'hash_train.npy'), trainList)
    print 'End Hash: Speed Time %f' % (time.time() - m_t)

def resetRandom():
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
    np.save(os.path.join(path, test_name + '_test.npy'), tempList[:int(num * test_ratio)])
    np.save(os.path.join(path, test_name + '_train.npy'), tempList[int(num * test_ratio):])

def spliteAllOfPath():
    subfolders = [folder for folder in os.listdir(
        path) if os.path.isdir(os.path.join(path, folder))]
    print subfolders
    for imgDir in subfolders:
        t_time = time.time()
        print 'Start ImageDir: %s ' % os.path.join(path, imgDir)
        splits_resamples(os.path.join(path, imgDir), tilesPerImage = tilesPerImage)
        print 'End ImageDir: %s Speed Time: %f' % (os.path.join(path, imgDir), (time.time() - t_time))

def runTest():
    m_bad = 0
    m_right = 0
    m_num = 0
    for main_times in range(0, times):
        if resetTest:
            resetRandom()
        test = np.load(os.path.join(path, test_name + '_test.npy'))
        train = np.load(os.path.join(path, test_name + '_train.npy'))
        testNum = len(test)
        trainNum = len(train)
        m_t = time.time()
        # logging.info('Start test: %d  train: %d' % (testNum, trainNum))
        for i in test:
            t1 = time.time()
            minD = []
            tempI = np.ravel(i[0])
            for j in train:
                tempJ = np.ravel(j[0])
                dist = getDistances(tempI, tempJ, type=distType)
                if is_big_key:
                    minD.append([dist, j[1], j[2], ks[j[1]]])
                else:
                    minD.append([dist, j[1], j[2]])
            temp = getMinOfNum(minD, k)
            is_true = False
            for l in temp:
                if is_big_key:
                    if l[3] == ks[i[1]]:
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
                        logging.error('###### Bad %s(%s: %s) with %s' (ks[i[1]], i[1], i[2], temp))
                    else:
                        logging.error('###### Bad %s: %s with %s' (i[1], i[2], temp))
            m_num += 1
            if m_num % reportTime == 1:
                logging.info('Last accuracy: %.2f %%' % (m_right / float(m_num) * 100.0))
                logging.info('Last loss: %.2f %%' % (m_bad / float(m_num) * 100.0))
                logging.info('right: %d bad: %d now: %d/%d Time: %.2fs/iter' % (m_right, m_bad, m_num, testNum * times, (time.time() - t1)))
        # logging.info('End test: %d  train: %d  %f s' % (testNum, trainNum, (time.time() - m_t)))
    logging.info('Last accuracy: %.2f %%' % (m_right / float(m_num) * 100.0))
    logging.info('Last loss: %.2f %%' % (m_bad / float(m_num) * 100.0))
    logging.info('End Run Test')

if __name__ == '__main__':
    import sys
    import getopt
    from collections import Counter
    import random

    opts, args = getopt.getopt(sys.argv[1:], 'f:sltzr:ai:mk:gx:v:hb', ['time=', 'dist=', 'report=', 'hash', 'size', 'log', 'round='])
    for op, value in opts:
        if op == '-f':
            path = value
        elif op == '-h':
            resetTest = True
        elif op == '-v':
            test_name = value
        elif op == '-g':
            loadFeature()
        elif op == '--log':
            is_log = True
        elif op == '--round':
            num_round = int(value)
        elif op == '-b':
            is_big_key = True
            subfolders = [folder for folder in os.listdir(
                path) if os.path.isdir(os.path.join(path, folder))]
            for file in subfolders:
                print 'Start %s' % file
                path2 = os.path.join(path, file)
                subfolders2 = [folder for folder in os.listdir(
                    path2) if os.path.isdir(os.path.join(path2, folder))]
                for file2 in subfolders2:
                    if ks.has_key(file2):
                        print '######### Error Has Same: %s(%s) %s' % (file, file2, ks[file2])
                    ks[file2] = file
                print 'End %s' % file
        elif op == '--hash':
            loadHash()
        elif op == '-x':
            mxnetpath = value
            sys.path.insert(0, mxnetpath)
        elif op == '-k':
            k = int(value)
        elif op == '-z':
            not_double = False
        elif op == '-m':
            removeAllSpliteOfPath()
        elif op == '-i':
            tilesPerImage = int(value)
        elif op == '--time':
            times = int(value)
        elif op == '--dist':
            distType = int(value)
        elif op == '--report':
            reportTime = int(value)
        elif op == '-a':
            spliteAllOfPath()
        elif op == '-r':
            test_ratio = float(value)
        elif op == '-s':
            load_all_img(path, not_double=not_double)
        elif op == '--size':
            test = np.load(os.path.join(path, test_name + '_test.npy'))
            train = np.load(os.path.join(path, test_name + '_train.npy'))
            print 'Size: %d' % (len(test) + len(train))
        elif op == '-l':
            test, train = load_all_beOne(path, test_ratio=test_ratio)
            np.save(os.path.join(path, 'knn_test.npy'), test)
            np.save(os.path.join(path, 'knn_train.npy'), train)
        elif op == '-t':
            runTest()