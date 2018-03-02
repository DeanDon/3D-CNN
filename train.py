#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: guohao_dong
@license: (C) Copyright 2016-2017, AccuRad AI.
@contact: guohao_dong@accurad.com
@software: AI
@file: UCF101_img_pro.py.py
@time: 2017/10/11 8:50
@desc:
'''
import keras
import cv2
import os
import numpy as np
import random
from config import cfg
from keras.applications import VGG16
import time
#from keras.preprocessing import image
#import tensorflow as tf
#import keras.backend.tensorflow_backend as K
#list_trainsets=['trainlist01.txt','trainlist02.txt','trainlist03.txt']
list_trainsets=['./ucfTrainTestlist/trainlist01.txt','./ucfTrainTestlist/trainlist02.txt','./ucfTrainTestlist/trainlist03.txt']
list_train_file=[]
for trainset in list_trainsets:
    for line in open(trainset).readlines():
        list_train_file.append(line.strip().split(' '))

def symbol(num_class=101,lr_=0.1):
    model = keras.models.Sequential()
    model = keras.models.Sequential()
    model.add(keras.layers.Conv3D(filters=32,
                                  kernel_size=(3, 3, 3),
                                  input_shape=(32, 64, 64, 3),
                                  padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.Conv3D(filters=32,
                                  kernel_size=(3, 3, 3),
                                  padding='same'
                                  ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.MaxPool3D(pool_size=(2, 2, 2)))
    model.add(keras.layers.Dropout(rate=0.25))

    model.add(keras.layers.Conv3D(filters=64,
                                  kernel_size=(3, 3, 3),
                                  padding='same'
                                  ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.Conv3D(filters=64,
                                  kernel_size=(3, 3, 3),
                                  padding='same'
                                  ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.MaxPool3D(pool_size=(2, 2, 2)))
    model.add(keras.layers.Dropout(rate=0.25))

    model.add(keras.layers.Conv3D(filters=128,
                                  kernel_size=(3, 3, 3),
                                  padding='same'
                                  ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.Conv3D(filters=128,
                                  kernel_size=(3, 3, 3),
                                  padding='same'
                                  ))
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.Conv3D(filters=128,
                                  kernel_size=(3, 3, 3),
                                  padding='same'
                                  ))
    model.add(keras.layers.BatchNormalization())


    #model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.MaxPool3D(pool_size=(2, 2, 2)))
    model.add(keras.layers.Dropout(rate=0.25))
    print model.output_shape

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate=0.50))

    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate=0.50))

    model.add(keras.layers.Dense(num_class, activation='softmax'))

    sgd = keras.optimizers.SGD(lr=lr_, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss=keras.losses.mean_squared_error, metrics=['accuracy'])
    return model
def batch_data_w(batch_size):
    while 1:
        random.shuffle(list_train_file)
        video_list = []
        label_list = []
        index = 0
        while 1:
            train_file = list_train_file[index]
            index += 1
            path = os.path.join(cfg.DATA_DIR, train_file[0])
            num_frame = len(os.listdir(path))
            try:
                start_frame = random.randint(0, num_frame - 32)
            except:
                continue
            frame_list = []
            for i in range(start_frame, start_frame + 32):
                frame_list.append(cv2.resize(cv2.imread(path + '/' + str(i) + '.jpg'), (64, 64)))
            frame_ndarray = np.array(frame_list)
            video_list.append(frame_ndarray)
            label_list.append(int(train_file[1]) - 1)
            if len(video_list) == batch_size:
                video_ndarray = np.array(video_list)
                label_ndarray = np.array(label_list)
                break
        yield video_ndarray, keras.utils.to_categorical(label_ndarray, num_classes=101)
def main():
    base_lr=5.0
    #model_3d = symbol(num_class=101)
    for i in range(10):
        #if i==0:
        #    continue
        model_3d = symbol(num_class=101,lr_=base_lr)
        #base_lr=base_lr/2
        if i!=0:
            model_3d.load_weights('Conv3D_101_8l'+str(i-1)+'.h5')
        #model_3d = symbol(num_class=101)
        model_3d.fit_generator(batch_data_w(16),steps_per_epoch=500,epochs=40)
        model_3d_json = model_3d.to_json()
        open('Conv3D.json', 'wb').write(model_3d_json)
        model_3d.save_weights('Conv3D_101_8l'+str(i)+'.h5')
        time.sleep(1000)
main()
