from multiprocessing.connection import Client
import sys
import getopt
import traceback
sys.path.insert(0, '/home/lol/anaconda2/lib/python2.7/site-packages')
import falconn
import os
import cv2
import numpy as np
import time

if __name__ == '__main__':
    filepath = None
    opts, args = getopt.getopt(sys.argv[1:], 'f:')
    for op, value in opts:
        if op == '-f':
            filepath = value
    if filepath is None:
        print 'Must Use -f set Img Path'
    else:
        c = Client('./server.temp', authkey=b'lee123456')
        c.send(['-f', filepath])
        print c.recv()