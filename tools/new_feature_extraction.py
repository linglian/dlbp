#coding=utf-8
import numpy as np
import cv2
import sys
import getopt
sys.path.insert(0, '/home/lol/dl/mxnet/python')
import mxnet as mx
import os
import shutil
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

def checkFold(name):
    if not os.path.exists(name):
        os.mkdir(name)

def removeDir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)

def removeFile(name):
    if os.path.exists(name):
        os.remove(name)

def splits_resamples(facescrub_root):
    #import sklearn
    import numpy as np
    #import matplotlib.pyplot as plt
    #import skimage
    #from skimage import transform as tf
    import shutil
    # this file is expected to be in {caffe_root}/examples/siamese
    # facescrub_root = '/home/lol/dl/image'
    thresholdGLOABL = 0.42
    import sys
    import os
    import Image
    import random
    import math

    fold = facescrub_root
    print fold
    removeDir(fold + '/examples')

    subfolders = [folder for folder in os.listdir(
        facescrub_root) if os.path.isdir(os.path.join(facescrub_root, folder))]
    print subfolders

    dict = {}
    for subfolder in subfolders:
        removeFile(os.path.join(facescrub_root, subfolder, 'test.npy'))
        removeFile(os.path.join(facescrub_root, subfolder, 'train.npy'))
        imgsfiles = [os.path.join(facescrub_root, subfolder, img)
                     for img in os.listdir(os.path.join(facescrub_root, subfolder))]
        for img in imgsfiles:
            dict[img] = subfolder
    print dict

    def im_crotate_image_square(im, deg):
        im2 = im.rotate(deg, expand=1)
        im = im.rotate(deg, expand=0)

        width, height = im.size
        assert (width == height)

        rads = math.radians(deg)
        new_width = width - (im2.size[0] - width)

        left = top = int((width - new_width) / 2)
        right = bottom = int((width + new_width) / 2)

        return im.crop((left, top, right, bottom))

    tilesPerImage = 360

    dx = dy = 224
    fold_idx = 1
    rotateAction = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
                    Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

    rotate45degree = [45, 135, 270]
    subfolders = [folder for folder in os.listdir(
        fold) if os.path.isdir(os.path.join(fold, folder))]
    print 'files: %s' % subfolders

    checkFold(fold + '/examples')

    for subfolder in subfolders:
        imgsfiles = [os.path.join(fold, subfolder, img)
                     for img in os.listdir(os.path.join(fold, subfolder)) if img.endswith('.JPG')]
        print 'Start Directory: %s' % subfolder
        for imgfile in imgsfiles:
            checkFold(fold + '/examples/' + subfolder)
            print 'Start Image: %s' % imgfile
            im = Image.open(imgfile)
            w, h = im.size
            im = im.crop((0, 0, w, int(h * 0.9)))
            #dx = 224
            for i in range(1, tilesPerImage + 1):
                newname = imgfile.replace('.', '_{:03d}.'.format(i))
                # print newname
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
                im_cropped = im.crop((x, y, x + dx, y + dy))
                if i % 2 == 0:  # roate 180,90
                    im_cropped = im_cropped.transpose(
                        random.choice(rotateAction))
                if i % 2 == 0 and i > 300:
                    roate_drgree = random.choice(rotate45degree)
                    im_cropped = im_crotate_image_square(
                        im_cropped, roate_drgree)
                newname = newname.replace(fold, fold + '/examples')
                im_cropped.save(newname)
            # don't remove startImg
            # os.remove(imgfile)
    return fold + '/examples'

def get_image(path):
    # download and show the image
    m = cv2.imread(path, 1)
    img = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # resize to 224*224 to fit model
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)  # change to (c, h,w) order
    img = img[np.newaxis, :]  # extend to (n, c, h, w)
    return img

def getFeatures(img, f_mod):
    img = get_image(img)
    f_mod.forward(Batch([mx.nd.array(img)]))
    features = f_mod.get_outputs()[0].asnumpy()
    return features

def init(GPUid=0):
    sym, arg_params,aux_params = mx.model.load_checkpoint('full-resnet-152',0)

    mod= mx.mod.Module(symbol=sym, context=mx.gpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
    mod.set_params(arg_params, aux_params, allow_missing=True)

    all_layers = sym.get_internals()
    f = all_layers['fc1_output']

    f_mod = mx.mod.Module(symbol=f, context=mx.gpu(GPUid), label_names=None)
    f_mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
    f_mod.set_params(arg_params, aux_params, allow_missing=True)

    return f_mod


def batch(folder, loadfolder, test_ratio = 0.02):
    import numpy as np
    import cv2
    import sys
    sys.path.insert(0, '/home/lol/dl/mxnet/python')
    import mxnet as mx
    import argparse
    import time
    import os
    import random
    import multiprocessing

    def handleFolder(GUPid, tasks):
        k = 0
        mod = init(GUPid)
        print 'Start tasks: %s' %tasks 
        for subfolder in tasks:
            workspace_folder = os.path.join(folder, subfolder)
            load_folder =  os.path.join(loadfolder, subfolder)
            print "extract label####", subfolder, "---GPU: ", GUPid, " process: ", k, "/", len(tasks)
            i = 0
            k += 1
            feature_array = []
            print 'Start: %s' % workspace_folder
            for filename in os.listdir(workspace_folder):
                # print 'Start : %s' % filename
                if '.jpg' in filename or '.JPG' in filename:
                    i += 1
                    f = getFeatures(os.path.join(workspace_folder, filename), mod)
                    feature_array.append((f[0], subfolder, filename))
                # print 'End : %s' % filename
            random.shuffle(feature_array)
            # print len(feature_array)
            print 'Save: %s/%s' % (load_folder, "test.npy")
            np.save((os.path.join(load_folder, "test.npy")),
                    feature_array[:int(i * test_ratio)])
            print 'Save: %s/%s' % (load_folder, "train.npy")
            np.save((os.path.join(load_folder, "train.npy")),
                    feature_array[int(i * (test_ratio)):])

    print 'Start'
    t1 = time.time()

    GPUvector = [0]
    nGPU = len(GPUvector)

    jobs = []
    subfolders = [fold for fold in os.listdir(folder)]
    fa = np.array_split(subfolders, nGPU)
    for i in range(nGPU):
        print "##################GPU ", str(i), " take", len(fa[i]), "folders"
    z = 0
    for i in GPUvector:
        mp_kwargs = dict(
            GUPid=i,
            tasks=fa[z]
        )
        p = multiprocessing.Process(target=handleFolder, kwargs=mp_kwargs)
        jobs.append(p)
        p.start()
        z += 1
        print "##################go to next index,start process for:", z, " process"
    for p in jobs:
        print "end of process"
        p.join()
    print "whole process time:", time.time() - t1


if __name__=='__main__':
    
    filePath = '/home/lol/dl/Image'

    import time

    opts, args = getopt.getopt(sys.argv[1:], 'f:sf:')
    for op, value in opts:
        if op == '-f':
            filePath = value
            sp = splits_resamples(filePath)
            batch(sp, filePath)
            removeDir(sp)
        elif op == '-sf':
            filePath = value

    subfolders = [folder for folder in os.listdir(
        filePath) if os.path.isdir(os.path.join(filePath, folder))]
    print subfolders

    for file in subfolders:
        t1 = time.time()
        path = os.path.join(filePath, file)
        print '$$$$$$$$$$$ Start: ', path, 'Begin Time: ', t1
        sp = splits_resamples(path)
        batch(sp, path)
        removeDir(sp)
        print '$$$$$$$$$$$ End: ', path, 'End Time: ', time.time()
        print 'Speed Time: ', time.time() - t1
    
    train_array = []
    test_array = []
    for file in subfolders:
        path = os.path.join(filePath, file)
        subfolders2 = [folder for folder in os.listdir(
            path) if os.path.isdir(os.path.join(path, folder))]
        print subfolders2
        for file2 in subfolders2:
            path2 = os.path.join(path, file2)
            test = np.load(os.path.join(path2, 'test.npy'))
            test_array.append(test)
            train = np.load(os.path.join(path2, 'train.npy'))
            train_array.append(train)
    np.save(os.path.join(filePath, 'train.npy'), train_array)
    np.save(os.path.join(filePath, 'test.npy'), test_array)

