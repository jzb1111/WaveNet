# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 22:11:41 2021

@author: asus
"""

import tensorflow as tf

def stretch(input_x,length):
    #restmp=tf.ones_like(input_x)
    #res=tf.reshape(input_x,[-1,])
    restmp=tf.expand_dims(input_x,1)
    restmp=tf.image.resize_images(restmp,(1,length))
    res=tf.squeeze(restmp,axis=1)
    return res