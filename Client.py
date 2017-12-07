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
    try:
        c = Client('./server.temp', authkey=b'lee123456')
        c.send(sys.argv[1:])
        ar = c.recv()
        for i in ar:
            print i
    except EOFError:
        print 'Connection closed, Please Reset Server.'