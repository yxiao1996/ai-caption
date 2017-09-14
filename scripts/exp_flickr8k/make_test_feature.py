import pandas as pd
import numpy as np
import os
import cPickle
from cnn_util import *

caffenet_model = '/home/xiaoy/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
caffenet_deploy = '/home/xiaoy/caffe/models/bvlc_reference_caffenet/deploy.prototxt'

feat_test_path = './feats_borg.npy'

# build test feature
test_images = ['./data/test_image/bjorn-borg.jpg']
cnn_test = CNN(model=caffenet_model, deploy=caffenet_deploy, width=227, height=227)
feats_test = cnn_test.get_features(test_images, layers='conv5', layer_sizes=[256,13,13])
np.save(feat_test_path, feats_test)
