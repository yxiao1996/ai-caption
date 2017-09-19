#-*- coding: utf-8 -*-
import math
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import cPickle
import json
import jieba as jb
import h5py
import baseline_model as base
from baseline_model import Caption_Generator
from keras.preprocessing import sequence

"""model parameters"""
n_epochs=1000
batch_size=50
dim_embed=256
dim_ctx=512
dim_hidden=256
ctx_shape=[196,512]
pretrained_model_path = './model/model-8'

"""file paths"""
anno_file = '/home/xiaoy/workspace/datasets/ai_challenger_caption/caption_validation_annotations_20170910.json'
feat_path = '/home/xiaoy/workspace/caption/feature_extract/feature/feature_vgg_16.npy'
model_path = '/home/xiaoy/workspace/caption/model/'
h5feats_path = '/home/xiaoy/workspace/caption/feature_extract/feature/test.h5'

"""hyper parameters"""
learning_rate = 0.001


def load_captions():
    # load annotations in json format
    annotations = json.load(open(anno_file, 'r'))
    # build caption list (parse sentense)
    space  = " "
    captions = []
    for anno in annotations:
    	for i in range(5):
            cap = anno['caption'][i]
            cap_cut = jb.cut(cap, cut_all = False)
            cap_join = space.join(cap_cut)
            captions.append(cap_join)
    return captions

def iter_read_feature(feats_path, chunksize=10):
    # load feature file
    with h5py.File(feats_path, 'r') as f:
        chunk = f['data'][:chunksize]
        chunk_index = 1        
        while len(chunk):
            current_feature = []
            for i in range(len(chunk)):
                for k in range(5):
                    current_feature.append(chunk[i])
            yield current_feature
            chunk = f['data'][chunksize*(chunk_index):chunksize*(chunk_index+1)]
            chunk_index += 1

def iter_read_caption(chunksize=50):
    # load captions
    captions = load_captions()
    chunk = captions[:chunksize]
    chunk_index = 1
    while len(chunk):
        yield chunk
        chunk = captions[chunksize*chunk_index:chunksize*(chunk_index+1)]
        chunk_index += 1

def train_baseline():
    # load captions
    captions = load_captions()
    # build vocabulary
    wordtoix, ixtoword, bias_init_vector = base.preProBuildWordVocab(captions)
    # number of unique word in annotation
    n_words = len(wordtoix)
    # maximum sentence length
    maxlen = np.max( map(lambda x: len(x.split(' ')), captions) )
    # load feature in npy format
    feats = np.load(feat_path)
    print "number of feature maps: ", len(feats)    
    # build feature dictionary
    image_id = []
    for i in range(len(feats)):
        for j in range(5):
            image_id.append(i)
    
    sess = tf.InteractiveSession()

    # build baseline model
    caption_generator = Caption_Generator(
            n_words=n_words,
            dim_embed=dim_embed,
            dim_ctx=dim_ctx,
            dim_hidden=dim_hidden,
            n_lstm_steps=maxlen+1, 
            batch_size=batch_size,
            ctx_shape=ctx_shape,
            bias_init_vector=bias_init_vector)
    loss, context, sentence, mask = caption_generator.build_model()
    saver = tf.train.Saver(max_to_keep=50)
    # load SGD optimizer
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    # run initialization
    tf.initialize_all_variables().run()
    for epoch in range(n_epochs):
    	for start, end in zip( \
                         range(0, len(feats), batch_size),
                         range(batch_size, len(feats), batch_size)):
            current_feats = feats[ image_id[start:end] ]
            current_feats = current_feats.reshape(-1, ctx_shape[1], ctx_shape[0]).swapaxes(1,2)
        
            current_captions = captions[start:end]
            current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.split(' ')[:-1] if word in wordtoix], current_captions)
            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen+1)
            current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
        
            nonzeros = np.array( map(lambda x: (x != 0).sum()+1, current_caption_matrix ))
            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1

            _, loss_value = sess.run([train_op, loss], feed_dict={
                context:current_feats,
                sentence:current_caption_matrix,
                mask:current_mask_matrix})
            print "Current Cost: ", loss_value, "Epoch: ", epoch
        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

def train_baseline_chunk():
    # load captions
    captions = load_captions()
    # build vocabulary
    wordtoix, ixtoword, bias_init_vector = base.preProBuildWordVocab(captions)
    # number of unique word in annotation
    n_words = len(wordtoix)
    # maximum sentence length
    maxlen = np.max( map(lambda x: len(x.split(' ')), captions) )
    
    sess = tf.InteractiveSession()

    # build baseline model
    caption_generator = Caption_Generator(
            n_words=n_words,
            dim_embed=dim_embed,
            dim_ctx=dim_ctx,
            dim_hidden=dim_hidden,
            n_lstm_steps=maxlen+1, 
            batch_size=batch_size,
            ctx_shape=ctx_shape,
            bias_init_vector=bias_init_vector)
    loss, context, sentence, mask = caption_generator.build_model()
    saver = tf.train.Saver(max_to_keep=50)
    # load SGD optimizer
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    # run initialization
    tf.initialize_all_variables().run()
    for epoch in range(n_epochs):
        # build feature & caption generator
        gen_feats = iter_read_feature(h5feats_path)
        gen_caption = iter_read_caption()
        batch_no = 0
        for current_feat in gen_feats:
            current_feats = np.array(current_feat)
            current_feats = current_feats.reshape(-1, ctx_shape[1], ctx_shape[0]).swapaxes(1,2)
        
            current_captions = next(gen_caption)
            current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.split(' ')[:-1] if word in wordtoix], current_captions)
            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen+1)
            current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
        
            nonzeros = np.array( map(lambda x: (x != 0).sum()+1, current_caption_matrix ))
            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1

            _, loss_value = sess.run([train_op, loss], feed_dict={
                context:current_feats,
                sentence:current_caption_matrix,
                mask:current_mask_matrix})
            print "Current Cost: ", loss_value, "Epoch: ", epoch, "Batch: ", batch_no
            batch_no += 1
        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
        
if __name__ == '__main__':
    train_baseline_chunk()
