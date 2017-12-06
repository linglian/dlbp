from multiprocessing.connection import Listener
import Queue
import sys
import getopt

path = './feature_train.npy'
mxnetpath = '/home/lol/dl/mxnet/python'
sys.path.insert(0, mxnetpath)
num_round = 0
prefix = "full-resnet-152"
layer = 'pool1_output'

def init(GPUid=0):
    import mxnet as mx
    model = mx.model.FeedForward.load(
        prefix, num_round, ctx=mx.gpu(GPUid), numpy_batch_size=1)
    internals = model.symbol.get_internals()
    fea_symbol = internals[layer]
    feature_extractor = mx.model.FeedForward(ctx=mx.gpu(GPUid), symbol=fea_symbol, numpy_batch_size=1,
                                             arg_params=model.arg_params, aux_params=model.aux_params, allow_extra_params=True)
    init_mod = feature_extractor
    return feature_extractor

if __name__=='__main__':
    task_queue = Queue.Queue()
    opts, args = getopt.getopt(sys.argv[1:], 'f:x:')
    for op, value in opts:
        if op == '-f':
            path = value
        elif op == '-x':
            mxnetpath = value
            sys.path.insert(0, mxnetpath)
    

