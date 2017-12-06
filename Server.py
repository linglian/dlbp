from multiprocessing.connection import Listener
import Queue
import sys
import getopt
import traceback
sys.path.insert(0, '/home/lol/anaconda2/lib/python2.7/site-packages')
import falconn
import os
import cv2
import numpy as np
import time


filepath = '/home/lol/dl/Image/feature_train.npy'
mxnetpath = '/home/lol/dl/mxnet/python'
sys.path.insert(0, mxnetpath)
num_round = 0
prefix = "full-resnet-152"
layer = 'pool1_output'
is_pool = True
dim = 2048


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
    train = np.load(filepath)
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
    tList = train[q.find_k_nearest_neighbors(fal, k)]
    return tList


def make_work(conn, mod, q, train):
    try:
        while True:
            msg = conn.recv()
            opts, args = getopt.getopt(msg, 'f:')
            img = None
            for op, value in opts:
                if op == '-f':
                    img = value
                elif op == '-z':
                    return 'Close'
            print getTest(img, mod, train, q, k=20)
    except EOFError:
        print 'Connection closed'
        return None


def run_server(address, authkey, mod, q, train):
    serv = Listener(address, authkey=authkey)
    while True:
        try:
            client = serv.accept()
            msg = make_work(client, mod, q, train)
            if msg == 'Close':
                break
        except Exception:
            traceback.print_exc()


if __name__ == '__main__':
    task_queue = Queue.Queue()
    opts, args = getopt.getopt(sys.argv[1:], 'f:x:')
    print sys.argv[0:]
    print sys.argv[1:]
    for op, value in opts:
        if op == '-f':
            filepath = value
        elif op == '-x':
            mxnetpath = value
            sys.path.insert(0, mxnetpath)
    mod, q, train = init()
    run_server('./server.temp', b'lee123456', mod, q, train)
