# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 22:05:30 2021

@author: asus
"""

import tensorflow as tf
from support.Config import config
from wavenet import WaveNet
from stretch import stretch
import numpy as np
from support.read_data import read_train_data

lr=0.001

input_x_=tf.placeholder(tf.float32,[None,None],name='input_x')
input_c_=tf.placeholder(tf.float32,[None,None,7],name='input_c')
binauray_label=tf.placeholder(tf.float32,[None,None,2])

stretch_length=tf.placeholder(tf.int32,name='stretch_length')
input_x=tf.expand_dims(input_x_,2)
input_c=stretch(input_c_,stretch_length)

WN=WaveNet(input_x,input_c)



init=tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)
    for i in range(500001):
        sjs1=np.random.randint(1,9)
        sjs2=np.random.randint(0,43200000)
        binaural,mono,rx,tx=read_train_data(sjs1,sjs2)
        