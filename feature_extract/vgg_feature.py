import os
import sys
import json
from PIL import Image
sys.path.append("/home/xiaoy/workspace/tools/models/slim")

import tensorflow as tf
slim = tf.contrib.slim
from PIL import Image
from nets import vgg
from nets.vgg import *
import numpy as np
import urllib2
import h5py

# paths
anno_file = '/home/xiaoy/workspace/datasets/ai_challenger_caption/caption_validation_annotations_20170910.json'
img_path = '/home/xiaoy/workspace/datasets/ai_challenger_caption/caption_validation_images_20170910/3cd32bef87ed98572bac868418521852ac3f6a70.jpg'
data_path = '/home/xiaoy/workspace/datasets/ai_challenger_caption/caption_validation_images_20170910/'
feat_path = '/home/xiaoy/workspace/caption/feature_extract/feature/feature_vgg_16.npy'

def extract_feature(img_path, split='dev', batch_size=10, layer=16):
    l = str(layer)
    checkpoint_file = '/home/xiaoy/workspace/models/vgg_' + l + '.ckpt'

    tf.reset_default_graph()

    # input jpg string, using decoder of tf.image.decode_jpeg() 
    # load image through tf.WholeFileReader()
    filenames = img_path
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)

    reader = tf.WholeFileReader()
    key, image_file = reader.read(filename_queue)
    
    # preprocess image 
    decode_jpeg = tf.image.decode_jpeg(image_file, channels=3)

    if decode_jpeg.dtype != tf.float32:
        decode_jpeg = tf.image.convert_image_dtype(decode_jpeg, dtype=tf.float32)
    # image = tf.expand_dims(decode_jpeg, 0)
    # scaled_input_tensor = tf.image.resize_bilinear(decode_jpeg, [224,224], align_corners=False)
    scaled_input_tensor = tf.image.resize_images(decode_jpeg, [224, 224])
    scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
    scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)
    
    # create image batch
    scaled_input_tensor.set_shape((224, 224, 3))
    num_preprocess_threads = 1

    images = tf.train.batch([scaled_input_tensor],
                            batch_size=batch_size,
                            num_threads=num_preprocess_threads
                            )


    #Load the model
    arg_scope = vgg_arg_scope()
    with slim.arg_scope(arg_scope):
        if layer == 11:
            logits, end_points = vgg_a(images, is_training=False)
        elif layer == 16:
            logits, end_points = vgg_16(images, is_training=False)
        elif layer == 19:
            logits, end_points = vgg_19(images, is_training=False)
        else:
            raise Exception("layer should be one of 11, 16, 19")
                

    feat_img = np.zeros([len(filenames), 196, 512])
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_file)

        print 'start'
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(len(filenames)/batch_size):
            predict_values, logit_values = sess.run([end_points['vgg_'+l+'/conv5/conv5_3'], logits])
            predict_values = np.array(predict_values)
            predict_values = np.reshape(predict_values, [-1, 196, 512])
            
            if i == 0:
                print predict_values.shape
                print np.array(logit_values).shape

            feat_img[i*batch_size:(i+1)*batch_size, :] = predict_values
            sys.stdout.write('\r %d/%d' %  (i, len(filenames)/batch_size))
            sys.stdout.flush()

        coord.request_stop()
        coord.join(threads)
        sess.close()
        
    # create output h5 file for training set.
    # f = h5py.File('img_' + split + '_vgg_' + l + '.h5', "w")
    # f.create_dataset(split + "_img", dtype='float32', data=feat_img)
    # f.close()
    return feat_img

def load_dataset():
    # extract file path from annotation file iin json format (30_000 images)
    annotations = json.load(open(anno_file, 'r'))
    # print annotations[0]

    # build image path list
    image_path = [data_path + str(anno['image_id']) for anno in annotations]
    # print image_path[:10]
    
    #im = Image.open(image_path[0]).resize((224,224))
    #im.show()
    return image_path

if __name__ == '__main__':
    image_path = load_dataset()
    feat = extract_feature(image_path[:1000], 'dev', 10)
    np.save(feat_path, feat)
    # check feature
    print(feat[0][0][0:49])
