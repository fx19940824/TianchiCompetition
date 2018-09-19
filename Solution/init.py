from TianchiCompetition.dataset import datamanager,clip_data
from TianchiCompetition.Solution.model import modelmanager
import keras
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.utils.generic_utils import Progbar
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,TensorBoard
import numpy as np
from collections import defaultdict
#def read_and_decode(filename_queue):

#def inputs(train, batch_size, num_epochs):
#    if not num_epochs:
#        num_epochs = None

#def convert_to(data_set,name):

def get_batch(image_path,label,image_W,image_H,batch_size,epochs,shuffle=False):
    input_queue=tf.train.slice_input_producer([image_path,label],shuffle=shuffle,num_epochs=epochs)
    
    image_contents=tf.read_file(input_queue[0])

    image=tf.image.decode_jpeg(image_contents,channels=3)

    image=tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)

    image=tf.image.per_image_standardization(image)
   
    image_batch=tf.train.batch([image],batch_size=batch_size,num_threads=4,capacity=batch_size*8)

    #label_batch=tf.reshape(label,[batch_size])

    return image_batch#,label_batch

def get_image(image_path,image_W,image_H,batch_size,epochs,shuffle=False):
    image_path=tf.cast(image_path,tf.string)

    input_queue=tf.train.slice_input_producer([image_path],shuffle=shuffle,num_epochs=epochs)
    
    image_contents=tf.read_file(input_queue[0])

    image=tf.image.decode_jpeg(image_contents,channels=3)

    image=tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)

    return image

if __name__=='__main__':
    #parameter
    batch_size=32
    inputshape=(192,256,3)
    epochs=50
    num_classes=12
    dirpath=os.path.split(os.path.realpath(__file__))[0]
    check_path=os.path.join(dirpath,'checkpoint/')
    log_path=os.path.join(dirpath,'logs/')

    #get dataset and information
    (x_train,y_train),(x_test,y_test)=datamanager.load_data()
    #image_batch=get_batch(x_train,y_train,192,256,batch_size,1)
    y_train=keras.utils.to_categorical(y_train,num_classes)

    model=modelmanager.Alexnet(input_shape=inputshape,num_classes=num_classes)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']    
    )
    model.summary()

    train_generator=datamanager.train_Generator(inputshape,batch_size)
    steps_per_epoch_train=len(x_train)/batch_size
    checkpoint=ModelCheckpoint(check_path,monitor='val_acc',verbose=0,save_best_only=False,mode='auto',save_weights_only=False)
    history=modelmanager.LossHistory()
    tensorboard=TensorBoard(log_dir=log_path,write_images=1,histogram_freq=0)
    callbacks_list=[checkpoint,history,tensorboard]

    validation_generator=datamanager.validation_Generator(inputshape,batch_size)

    model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch_train,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
        validation_data=validation_generator,
        validation_step=500
    )
    history.loss_plot('epoch')

    result=[]

    for i in range(len(x_test)):
        test_data=clip_data.clip_img(x_test[i])
        result.append(0)
        for test_batch in test_data:
            if datamanager.isDetected(model.predict(test_batch)):
                result[i]=1
                break
    
    print([result,y_test])

    '''train_history=defaultdict(list)
    test_history=defaultdict(list)
    for epoch in range(1,epochs+1):
        print('Epoch {}/{}'.format(epoch,epochs))
        num_batches=int(np.ceil(len(x_train)/float(batch_size)))
        progress_bar=Progbar(target=num_batches)

        epoch_loss=[]
        for index in range(num_batches):
            image_batch=get_image(x_train[index*batch_size:(index+1)*batch_size],192,256,batch_size,1,shuffle=True)
            label_batch=tf.convert_to_tensor(y_train[index*batch_size:(index+1)*batch_size])
            image_batch,label_batch=tf.Session().run([image_batch,label_batch])
            #label_batch=y_train[index*batch_size:(index+1)*batch_size]

            epoch_loss.append(model.train_on_batch(image_batch,label_batch))
            progress_bar.update(index+1)
        
        print('Testing for epoch {}:'.format(epoch))
        train_loss=np.mean(np.array(epoch_loss))
        train_history['trainloss'].append(train_loss)


        test_loss=model.evaluate(x_test,y_test)
        test_history['testloss'].append(test_loss)

        print('loss on training is {}'.format(train_loss))
        print('loss on test is {}'.format(test_loss))'''
    
    '''train_datagen=ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True
    )
    test_datagen=ImageDataGenerator(rescale=1./255)

    train_dir=datamanager.get_traindir()
    test_dir=datamanager.get_testdir()
    train_generator=train_datagen.flow_from_directory(
        train_dir,
        target_size=(256,256),
        batch_size=32
    )

    model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,

    )
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)

        try:
            while not coord.should_stop():
                for epoch in range(1,epochs+1):
                    print('Epoch {}/{}'.format(epoch,epochs))
                    num_batchs=int(np.ceil(x_train.shape[0]/float(batch_size)))
                    progress_bar=Progbar(target=num_batchs)


                    epoch_loss=[]
                    epoch_loss.append(model.train_on_batch())
                
                #imgs=get_batch(x_train,y_train,192,256,batch_size,1)
                
                sess.run(get_batch(x_train,y_train,192,256,batch_size,1))
                print(imgs.shape)
        except tf.errors.OutOfRangeError:
            print('done')
        finally:
            coord.request_stop()
        coord.join(threads)'''
        
    


    #print('x_train size:{}'.format(len(x_train)))
    #print('x_test size:{}'.format(len(x_test)))
   