#coding=utf-8

import os
import argparse
import logging
import sys
logging.basicConfig(level=logging.INFO)

prefix = "full-resnet-152"
num_round = 0
num_epoch = 5
mxnetPath = '/home/lol/dl/mxnet/python'
if __name__ == '__main__':
    import getopt
    opts, args = getopt.getopt(sys.argv[1:], 'x:p:r:e:')
    for op, value in opts:
        if op == '-x':
            mxnetPath = value
            sys.path.insert(0, '/home/lol/dl/mxnet/python')
        elif op == '-p':
            prefix = value
        elif op == '-r':
            num_round = value
        elif op == '-e':
            num_epoch = value

    sys.path.insert(0, mxnetPath)
    import mxnet as mx
    model = mx.model.FeedForward.load(
        prefix, epoch=num_round, ctx=mx.gpu(0), numpy_batch_size=1)
    internals = model.symbol.get_internals()


    data = internals["pool1_output"]

    print data

    fc1 = mx.symbol.FullyConnected(data=data, num_hidden=512, name='f1')

    relu1 = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")

    fc2 = mx.symbol.FullyConnected(data=fc1, num_hidden=8, name='f2')

    softmax = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')

    model = mx.model.FeedForward(ctx=mx.gpu(0), symbol=softmax, numpy_batch_size=1,
                                                arg_params=model.arg_params, aux_params=model.aux_params, allow_extra_params=True)

    model.num_epoch = num_epoch

    data_shape = (3, 224, 224)  

    train = mx.io.ImageRecordIter(  
        path_imgrec = 'train.rec',      
        data_shape  = data_shape,  
        batch_size  = 1,  
        rand_crop   = True,  
        rand_mirror = True)  

    val = mx.io.ImageRecordIter(  
        path_imgrec = 'test.rec',  
        rand_crop   = False,  
        rand_mirror = False,  
        data_shape  = data_shape,  
        batch_size  = 1)

    model.fit(
        X = train,
        eval_data = val)

    model.save('full-resnet-152')
    model.symbol.save('full-resnet-152')