from multiprocessing.connection import Listener
import sys
import getopt
import traceback
sys.path.insert(0, '/home/lol/anaconda2/lib/python2.7/site-packages')
import falconn
import os
import cv2
import numpy as np
import time
from PIL import Image

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='test_train_Server.log',
                    filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

path = '/home/lol/dl/Image/feature_train.npy'
mxnetpath = '/home/lol/dl/mxnet/python'
sys.path.insert(0, mxnetpath)
num_round = 0
prefix = "full-resnet-152"
layer = 'pool1_output'
is_pool = True
dim = 2048
reportTime = 500


def load_all_beOne(path):
    import time
    import random
    subfolders = [folder for folder in os.listdir(
        path) if os.path.isdir(os.path.join(path, folder))]
    print subfolders
    tt = time.time()
    main_imgArray = []
    per = 0
    print 'Start Merge Npy'
    n = 0
    testNum = 0
    for file in subfolders:
        filepath = os.path.join(path, file)
        print 'Start Merge Npy %s' % filepath
        subfolders2 = [folder for folder in os.listdir(
            filepath) if os.path.isdir(os.path.join(filepath, folder))]
        print subfolders2
        imgArray = []
        for file2 in subfolders2:
            t1 = time.time()
            filepath2 = os.path.join(filepath, file2)
            imgArray = np.load(os.path.join(filepath2, 'knn_splite.npy'))
            # print 'Load Knn.npy: %s' % (os.path.join(filepath2, knn_name + '.npy'))
            if len(imgArray) == 0:
                logging.error('Bad Npy: %s' %
                              os.path.join(filepath2, 'knn_splite.npy'))
                continue
            t_time = time.time()
            testNum += len(imgArray)
            for i in imgArray:
                main_imgArray.append(i)
                n += 1
                if n % reportTime == 0:
                    # print 'Finish %d/%d  SpeedTime: %f s' % (n, testNum, (time.time() - t_time))
                    t_time = time.time()
        print 'End Merge Npy: %d %f s' % (len(main_imgArray), (time.time() - tt))
    return main_imgArray


def getDistOfL2(form, to):
    return cv2.norm(form, to, normType=cv2.NORM_L2)

def checkFold(name):
    if not os.path.exists(name):
        os.mkdir(name)

def getImage(img):
    img = cv2.imread(img, 1)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = img[np.newaxis, :]
        return img
    else:
        return None


def getFeatures(img, f_mod=None):
    img = getImage(img)
    if img is not None:
        f = f_mod.predict(img)
        f = np.ravel(f)
        return f
    else:
        return None


def init_hash():
    train = load_all_beOne()
    trainNum = len(train)
    p = falconn.get_default_parameters(trainNum, dim)
    t = falconn.LSHIndex(p)
    dataset = [np.ravel(x[0]).astype(np.float32) for x in train]
    dataset = np.array(dataset)
    t.setup(dataset)
    if is_pool:
        q = t.construct_query_pool()
    else:
        q = t.construct_query_object()
    return (q, train)


def init_mxnet(GPUid=0):
    import mxnet as mx
    model = mx.model.FeedForward.load(
        prefix, num_round, ctx=mx.gpu(GPUid), numpy_batch_size=1)
    internals = model.symbol.get_internals()
    fea_symbol = internals[layer]
    feature_extractor = mx.model.FeedForward(ctx=mx.gpu(GPUid), symbol=fea_symbol, numpy_batch_size=1,
                                             arg_params=model.arg_params, aux_params=model.aux_params, allow_extra_params=True)
    init_mod = feature_extractor
    return feature_extractor


def init():
    mod = init_mxnet()
    q, train = init_hash()
    return (mod, q, train)


def getTest(img, mod, train, q, k=20):
    fal = getFeatures(img, f_mod=mod)
    if fal is not None:
        tList = train[np.array(q.find_k_nearest_neighbors(fal, k))]
        return fal, tList
    else:
        return fal, None


def make_work(conn, mod, q, train):
    try:
        while True:
            msg = conn.recv()
            logging.info(msg)
            opts, args = getopt.getopt(msg, 'f:zt:k:s', ['help'])
            img = None
            k = 20
            img_type = 0
            is_save = True
            msg = []
            for op, value in opts:
                if op == '-f':
                    img = value
                elif op == '-z':
                    return 'Close'
                elif op == '-k':
                    k = int(value)
                elif op == '-s':
                    is_save = False
                elif op == '-t':
                    img_type = int(value)
                elif op == '--help':
                    msg.append(' ')
                    msg.append('Usage:')
                    msg.append('  Client [options]')
                    msg.append(' ')
                    msg.append('General Options:')
                    msg.append('-f <path>\t\t Set test image path')
                    msg.append('-z \t\t\t Close server')
                    msg.append('-k <number>\t\t Set rank')
                    msg.append('-s \t\t\t No Save image of rank K')
                    msg.append('-t <number>\t\t Set image type if you want to know test type')
                    return msg
            if img is None:
                msg.append('Must set Image Path use -f')
                return msg
            
            ti_time = time.time()
            fal, tList = getTest(img, mod, train, q, k=k)
            msg.append('Test Image Spend Time: %.2lf s' % (time.time() - ti_time))
            is_Right = False
            if tList is None:
                 msg.append('Bad Img Path')
                 return msg
            else:
                ti_time2 = time.time()
                ti = str(int(time.time()))
                if is_save:
                    checkFold(os.path.join('/tmp/image'))
                    checkFold(os.path.join('/tmp/image', ti))
                    n = 0
                    m = cv2.imread(img, 1)
                    if m is not None:
                        im = cv2.resize(m, (1024, 1024))
                        im = Image.fromarray(im)
                        im.save('/tmp/image/%s/GT.JPG' % ti)
                    else:
                        msg.append('Bad Image: %s' % i[2])
                for i in tList:
                    if is_save:
                        n += 1
                        m = cv2.imread(i[2], 1)
                        if m is not None:
                            im = cv2.resize(m, (1024, 1024))
                            im = Image.fromarray(im)
                            im.save('/tmp/image/%s/%d_%d.JPG' % (ti, n, getDistOfL2(fal, i[0])))
                        else:
                            msg.append('Bad Image: %s' % i[2])
                    if img_type != 0 and img_type == int(i[1]):
                        is_Right = True
                if is_save:
                    msg.append('Save Image Spend Time: %.2lf s' % (time.time() - ti_time2))
                    msg.append('Save Image: /tmp/image/%s/' % ti)
            if is_Right and img_type != 0:
                msg.append('Find Right Image')
            elif img_type != 0:
                msg.append('Find Bad Image')
            return msg
    except EOFError:
        logging.info('Connection closed')
        return None


def run_server(address, authkey, mod, q, train):
    serv = Listener(address, authkey=authkey)
    while True:
        try:
            client = serv.accept()
            msg = make_work(client, mod, q, train)
            if msg == 'Close':
                break
            else:    
                client.send(msg)
        except Exception:
            traceback.print_exc()


if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], 'f:x:')
    print sys.argv
    for op, value in opts:
        if op == '-f':
            path = value
        elif op == '-x':
            mxnetpath = value
            sys.path.insert(0, mxnetpath)
    print 'Start Init'
    mod, q, train = init()
    print 'End Init'
    print 'Start Run'
    run_server('./server.temp', b'lee123456', mod, q, train)
    print 'Stop Run'
