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
        elif op == '-p':
            prefix = value
        elif op == '-r':
            num_round = value
        elif op == '-e':
            num_epoch = value

    sys.path.insert(0, mxnetPath)
    import mxnet as mx

    def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='pool1'):
        all_layers = symbol.get_internals()
        net = all_layers[layer_name+'_output']
        net = mx.symbol.FullyConnected(data=net, num_hidden=512, name='fc1')
        net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc2')
        net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
        return net

    def get_iterators(batch_size, data_shape=(3, 224, 224)):
        train = mx.io.ImageRecordIter(
            path_imgrec         = 'train.rec',
            data_name           = 'data',
            label_name          = 'softmax_label',
            batch_size          = batch_size,
            data_shape          = data_shape,
            shuffle             = True,
            rand_crop           = True,
            rand_mirror         = True)
        val = mx.io.ImageRecordIter(
            path_imgrec         = 'test.rec',
            data_name           = 'data',
            label_name          = 'softmax_label',
            batch_size          = batch_size,
            data_shape          = data_shape,
            rand_crop           = False,
            rand_mirror         = False)
        return (train, val)
    
    def fit(symbol, train, val, batch_size, num_gpus):
        devs = [mx.gpu(i) for i in range(num_gpus)]
        mod = mx.mod.Module(symbol=symbol, context=devs)
        opt = mx.optimizer.Adam(learning_rate=0.001)
        mult_dict = {k:0.0 for k in arg_params if not 'relu1' in k and not 'pool1' in k and not 'fc1' in k and not 'fc2' in k}
        opt.set_lr_mult(mult_dict)
        mod.fit(train, val,
            num_epoch=1,
            allow_missing=True,
            batch_end_callback = mx.callback.Speedometer(batch_size, 10),
            kvstore='device',
            optimizer=opt,
            optimizer_params={'learning_rate':0.0001},
            initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
            eval_metric='acc')
        mod.symbol.save('abc')
        metric = mx.metric.Accuracy()
        return mod.score(val, metric)

    data_shape = (3, 224, 224)
    num_classes = 8
    batch_per_gpu = 1
    num_gpus = 1
    batch_size = batch_per_gpu * num_gpus

    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, 0)
    new_sym = get_fine_tune_model(sym, arg_params, num_classes)
    (train, val) = get_iterators(batch_size)
    mod_score = fit(new_sym, train, val, batch_size, num_gpus)
    
    assert mod_score > 0.77, "Low training accuracy."


    # mod.save('full-resnet-152')
    # mod.symbol.save('full-resnet-152')