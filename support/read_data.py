# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 19:09:51 2021

@author: asus
"""

import soundfile
import sys
sys.path.extend(['E:\\deeplearning\\wave_net_binaural','E:/deeplearning/wave_net_binaural'])
from support.Config import config
import numpy as np

def read_wave_data(filename):
    wave_data,framerate=soundfile.read(filename)
    return wave_data#, framerate

def read_txt_file(filename):
    with open(filename, "r") as f:    #打开文件
        data = f.read()   #读取文件
        f.close()
    return data

def read_txt_data(filename):
    txt=read_txt_file(filename)
    strtmp=''
    res=[]
    restmp=[]
    for i in txt:
        if i!='\n' and i!=' ':
            strtmp+=i
        else:
            if i==' ':
                restmp.append(float(strtmp))
                strtmp=''
            if i=='\n':
                #re.append()
                restmp.append(float(strtmp))
                res.append(restmp)
                restmp=[]
                strtmp=''
    return res
            
def read_train_data(num,num2):
    
    binaural=read_wave_data('./binaural_dataset/trainset/subject'+str(num)+'/binaural.wav')
    mono=read_wave_data('./binaural_dataset/trainset/subject'+str(num)+'/mono.wav')
    rx=read_txt_data('./binaural_dataset/trainset/subject'+str(num)+'/rx_position.wav')
    tx=read_txt_data('./binaural_dataset/trainset/subject'+str(num)+'/tx_position.wav')
    
    if num2>len(binaural)-config['batch_size']*config['data_length']-1:
        num2=len(binaural)-config['batch_size']*config['data_length']-1
        
        
    binaural_sig=np.zeros((config['batch_size'],config['data_length'],2))
    mono_sig=np.zeros((config['batch_size'],config['data_length']))
    rx_=np.zeros((config['batch_size'],config['data_length']//400,7))
    tx_=np.zeros((config['batch_size'],config['data_length']//400,7))
    
    for i in range(config['batch_size']):
        for j in range(len(config['data_length'])):
            binaural_sig[i][j]=binaural[i*config['data_length']+j+num2]
            mono_sig[i][j]=mono[i*config['data_length']+j+num2]
            
            rx_[i][j]=rx[(i*config['data_length']+j+num2)//400]
            tx_[i][j]=tx[(i*config['data_length']+j+num2)//400]
    return binaural_sig,mono_sig,rx_,tx_