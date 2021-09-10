# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 16:08:16 2021

@author: asus
"""

import tensorflow as tf
from support.Config import config

class WaveNet():
    def __init__(self,input_x,input_c):
        self.input_x=input_x
        self.input_c=input_c
        
    def wave_net_block(self,input_x,filters,filter_size,dilation_rate):
        residual=input_x
        tanh_out=tf.layers.conv1d(inputs=input_x,filters=filters,kernel_size=filter_size,padding='same',dilation_rate=dilation_rate,activation=tf.nn.tanh)
        sigmoid_out=tf.layers.conv1d(inputs=input_x,filters=filters, kernel_size=filter_size,padding='same',dilation_rate=dilation_rate,activation=tf.nn.sigmoid)
        merge=tanh_out*sigmoid_out
        skip_out=tf.layers.conv1d(merge, filters, 1,activation=tf.nn.relu)
        out=skip_out+residual
        return out,skip_out
    
    def create_network(self):
        input_x=self.input_x
        input_c=self.input_c
        
        out_x,skip_x=self.wave_net_block(input_x,64,2,2)
        out_c,skip_c=self.wave_net_block(input_c,64,2,2)
        
        out=out_x+out_c
        skip=skip_x+skip_c
        
        skip_connection=skip
        for i in range(20):
            out,skip=self.wave_net_block(input_x,64,2,dilation_rate=2**((i+2)%9))
            skip_connection+=skip
        net=tf.nn.relu(skip_connection)
        net=tf.layers.conv1d(net,32,2,padding='same',activation=tf.nn.relu)
        net=tf.layers.conv1d(net,2,1,padding='same')
        #net=tf.layers.flatten(net)
        #net=tf.layers.fully_connected(net)
        return net
    
    