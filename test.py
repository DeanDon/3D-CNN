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
import tensorflow as tf
import keras.backend.tensorflow_backend as K
def keras_bakend_config(gpuid):
    """
    配置keras后端的计算资源使用
    :param gpuid:
    :return: None
    """
    _gpu_options = tf.GPUOptions(allow_growth=True,
                                 per_process_gpu_memory_fraction=0.3,
                                 visible_device_list=gpuid)

    if not os.environ.get('OMP_NUM_THREADS'):
        config = tf.ConfigProto(allow_soft_placement=True,
                                gpu_options=_gpu_options)
    else:
        num_thread = int(os.environ.get('OMP_NUM_THREADS'))
        config = tf.ConfigProto(intra_op_parallelism_threads=num_thread,
                                allow_soft_placement=True,
                                gpu_options=_gpu_options)
    _SESSION = tf.Session(config=config)
    K.set_session(_SESSION)
keras_bakend_config('1')
#from keras.preprocessing import image
#import tensorflow as tf
#import keras.backend.tensorflow_backend as K
#list_trainsets=['trainlist01.txt','trainlist02.txt','trainlist03.txt']
list_trainsets=['./ucfTrainTestlist/testlist01.txt','./ucfTrainTestlist/testlist02.txt','./ucfTrainTestlist/testlist03.txt']
list_train_file=[]
for trainset in list_trainsets:
    for line in open(trainset).readlines():
        list_train_file.append(line.strip().split(' '))

list_testsets=['./ucfTrainTestlist/testlist01.txt','./ucfTrainTestlist/testlist02.txt','./ucfTrainTestlist/testlist03.txt']
list_test_file=[]
for testset in list_testsets:
    for line in open(testset).readlines():
        list_test_file.append(line.strip())

cls_dict={}
for line in open('./ucfTrainTestlist/classInd.txt').readlines():
    line_split=line.strip().split(' ')
    cls_dict[line_split[1]]=int(line_split[0])-1
def symbol_9l(num_class=101,lr_=0.1):
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


    model.add(keras.layers.BatchNormalization())
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
    model.add(keras.layers.Dropout(rate=0.25))

    model.add(keras.layers.Dense(num_class, activation='softmax'))

    sgd = keras.optimizers.SGD(lr=lr_, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss=keras.losses.mean_squared_error, metrics=['accuracy'])
    return model
def symbol(num_class=101,lr_=0.1):
    model = keras.models.Sequential()
    model = keras.models.Sequential()
    model.add(keras.layers.Conv3D(filters=32,
                                  kernel_size=(3, 3, 3),
                                  input_shape=(32, 64, 64, 3),
                                  padding='same'))
    #model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.Conv3D(filters=32,
                                  kernel_size=(3, 3, 3),
                                  padding='same'
                                  ))
    #model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.MaxPool3D(pool_size=(1, 2, 2)))
    model.add(keras.layers.Dropout(rate=0.25))

    model.add(keras.layers.Conv3D(filters=64,
                                  kernel_size=(3, 3, 3),
                                  padding='same'
                                  ))
    #model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.Conv3D(filters=64,
                                  kernel_size=(3, 3, 3),
                                  padding='same'
                                  ))
    #model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.MaxPool3D(pool_size=(2, 2, 2)))
    model.add(keras.layers.Dropout(rate=0.25))

    model.add(keras.layers.Conv3D(filters=128,
                                  kernel_size=(3, 3, 3),
                                  padding='same'
                                  ))
    #model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.Conv3D(filters=128,
                                  kernel_size=(3, 3, 3),
                                  padding='same'
                                  ))
    #model.add(keras.layers.BatchNormalization())


    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.MaxPool3D(pool_size=(2, 2, 2)))
    model.add(keras.layers.Dropout(rate=0.25))
    print model.output_shape

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate=0.25))

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
    #model_3d = symbol(num_class=101)
    for i in range(10):
        model_3d = symbol(num_class=101,lr_=1.5*((0.5)**(i/2)))
        if i!=0:
            model_3d.load_weights('Conv3D_'+str(i-1)+'.h5')
        #model_3d = symbol(num_class=101)
        model_3d.fit_generator(batch_data_w(32),steps_per_epoch=500,epochs=10)
        model_3d_json = model_3d.to_json()
        open('Conv3D.json', 'wb').write(model_3d_json)
        model_3d.save_weights('Conv3D_'+str(i)+'.h5')
def test_data_batch():
    random.shuffle(list_test_file)
    #model=symbol(101)
    frame_list = []
    for test_file in list_test_file:
        path = os.path.join(cfg.DATA_DIR, test_file)
        cls_name=test_file.split('/')[0]
        list_frame=os.listdir(path)
        frame_list = []
        for frame in list_frame:
            frame_list.append(cv2.resize(cv2.imread(path+'/'+frame),(64,64)))
            if len(frame_list)==32:
                frame_array= np.array(frame_list)
                frame_list=[]
                yield frame_array,cls_dict[cls_name],cls_name
def train_data_batch():
    #random.shuffle(list_train_file)
    #model=symbol(101)
    frame_list = []
    num_set=0
    for train_file in list_train_file:
        path = os.path.join(cfg.DATA_DIR, train_file[0])
        cls_name=train_file[0].split('/')[0]
        num_frame=len(os.listdir(path))
        frame_list = []
        for frame in range(num_frame):
            frame_list.append(cv2.resize(cv2.imread(path+'/'+str(frame)+'.jpg'),(64,64)))
            if len(frame_list)==32:
                frame_array= np.array(frame_list)
                frame_list=[]
                #np.save(str(num_set)+'.npy',frame_array)
                num_set+=1
                yield frame_array,int(train_file[1])-1,cls_name
def pre_test():
    model=symbol_9l(101)
    model.load_weights('Conv3D_101_8l3.h5_92%')
    acc=0.0
    top_5_acc=0.0
    num_test_set=0
    for test_data,test_label,cls_name in test_data_batch():
        test_data=np.reshape(test_data,(1,32,64,64,3))
        softmax_out=model.predict(test_data,batch_size=1)
        if np.argmax(softmax_out)==test_label:
            acc+=1.0
        num_test_set+=1
        #print softmax_out
        #print cls_name,np.argmax(softmax_out),test_label
        print acc/num_test_set
pre_test()
#main()
