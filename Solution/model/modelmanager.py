import tensorflow as tf
import keras
from keras.models import Model,Sequential
from keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dense,Dropout
import matplotlib.pyplot as plt

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

def LeNet_5(input_shape=(32,32,1),num_classes=10):
    model=Sequential()
    model.add(Conv2D(6,5,padding='valid',activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(16,5,padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(120,activation='tanh'))
    model.add(Dense(84,activation='tanh'))

    input=Input(input_shape)
    features=model(input)
    output=Dense(num_classes,activation='softmax')(features)

    return Model(inputs=input,outputs=output)

def Alexnet(input_shape=(227,227,3),num_classes=5):
    cnn=Sequential()
    cnn.add(Conv2D(96,11,padding='valid',strides=4,input_shape=input_shape))
    cnn.add(MaxPooling2D(pool_size=2))
    cnn.add(Conv2D(256,5,padding='same'))
    cnn.add(MaxPooling2D(pool_size=2))
    cnn.add(Conv2D(384,3,padding='same'))
    cnn.add(Conv2D(384,3,padding='same'))
    cnn.add(Conv2D(256,3,padding='same'))
    cnn.add(Flatten())
    cnn.add(Dense(4096,activation='relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(4096,activation='relu'))
    cnn.add(Dropout(0.5))

    input=Input(input_shape)
    features=cnn(input)
    output=Dense(num_classes,activation='softmax')(features)
    return Model(inputs=input,outputs=output)