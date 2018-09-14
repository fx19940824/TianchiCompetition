import tensorflow as tf
import numpy as np
import keras
from collections import defaultdict
from keras.datasets import mnist,boston_housing,cifar100,fashion_mnist
from keras.layers import Input
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.models import Model,Sequential
from keras import optimizers
from keras.utils.generic_utils import Progbar
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import Dense,GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.misc import imresize

import os
from PIL import Image
from sklearn.model_selection import train_test_split

#read image
def read_image(imageName):
    im = Image.open(imageName).convert('RGB')
    data = np.array(im)
    return data

def load_data(input_dir):
    image_dir=os.listdir(input_dir)
    images_path=[]
    labels=[]
    for image_path in image_dir:
        for fn in os.listdir(os.path.join(input_dir, image_path)):
            if fn.endswith('.jpg'):
                fd = os.path.join(input_dir, image_path, fn)
                images_path.append(fd)
                labels.append(image_path)

    images=[]
    #for image_path in images_path:
    #images.append(read_image(image_path))
    images.append(read_image(images_path[0]))

    x=np.array(images)
    y=np.array(list(map(int,labels)))
    y=y[0]
    return x,y

def MyModel(inputshape,num_classes=10):
    base_model=InceptionV3(weights='imagenet',
                        #weights=None,
                        input_shape=inputshape,
                        include_top=False
                        )

    #for layer in base_model.layers:
    #    layer.trainable=False

    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x)

    predictions=Dense(num_classes,activation='softmax')(x)

    return Model(inputs=base_model.input,outputs=predictions)

def setup_to_transfer_learning(model,base_model):
    for layer in base_model.layers:
        layer.trainable=False

    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy']
                )

def setup_to_fine_tune(model,base_model):
    trainlayer=17
    for layer in base_model.layers[:trainlayer]:
        layer.trainable=False
    for layer in base_model.layers[trainlayer+1:]:
        layer.trainable=True
    
    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy']
                )
    

if __name__=='__main__':
    #parameter
    input_dir='D:/Data/tianchi/guangdong_round1_train1_20180903/'
    epochs=10
    num_classes=1
    resize_shape=(150,150)
    batch_size=128

    x_data,y_data=load_data(input_dir)
    x_train=x_test=x_data
    #y_data=keras.utils.to_categorical(y_data,num_classes)
    y_data=np.expand_dims(y_data,axis=-1)
    y_train=y_test=y_data

    #(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
    """(x_train,y_train),(x_test,y_test)=cifar100.load_data()
    x_train_reshape=[]
    x_test_reshape=[]
    num_trainsize=int(x_train.shape[0]/10)
    num_testsize=int(x_test.shape[0]/10)
    for i in range(num_trainsize):
        x_train_reshape.append(ndimage.zoom(x_train[i],(5,5,1)))
    x_train_reshape=np.array(x_train_reshape)
    y_train=y_train[0:num_trainsize]
    for i in range(num_testsize):
        x_test_reshape.append(ndimage.zoom(x_test[i],(5,5,1)))
    x_test_reshape=np.array(x_test_reshape)
    y_test=y_test[0:num_testsize] """

    model=MyModel(x_train[0].shape,num_classes)
    model.compile(optimizer='rmsprop',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
                )
    model.summary()

    model.fit(x_train,y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test,y_test)
            )
    
    score=model.evaluate(x_test,y_test,verbose=0)
    print('Test loss:',score[0])
    print('Test accuracy:', score[1])
    
    """ train_loss=[]
    test_loss=[]
    for epoch in range(1,epochs+1):
        print('Epoch  {}/{}'.format(epoch,epochs))
        curtrainloss=model.fit(x_train,y_train)
        train_loss.append(train_loss)
        curtestloss=model.fit(x_test,y_test)
        test_loss.append(curtestloss)
        print('loss on training {}'.format(curtrainloss))
        print('loss on test {}'.format(curtestloss)) """

    