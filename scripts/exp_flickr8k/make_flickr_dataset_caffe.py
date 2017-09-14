import pandas as pd
import numpy as np
import os
import cPickle
from cnn_util import *

flickr_image_path = './data/Flickr8k_Dataset/Flicker8k_Dataset'
feat_path = './data/feats_caffe.npy '
annotation_path = './data/Flickr8k.token.txt'
annotation_result_path = './data/annotations.pickle'

# use alexnet to extract feature
caffenet_model = '/home/xiaoy/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
caffenet_deploy = '/home/xiaoy/caffe/models/bvlc_reference_caffenet/deploy.prototxt'

cnn = CNN(model=caffenet_model, deploy=caffenet_deploy, width=227, height=227)

annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path,x.split('#')[0]))

unique_images = annotations['image'].unique()
image_df = pd.DataFrame({'image':unique_images, 'image_id':range(len(unique_images))})

annotations = pd.merge(annotations, image_df)

feats = cnn.get_features(unique_images, layers='conv5', layer_sizes=[256,13,13])
np.save(feat_path, feats)
