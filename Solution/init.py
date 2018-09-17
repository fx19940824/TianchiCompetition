from TianchiCompetition.dataset import datamanager
import tensorflow as tf

if __name__=='__main__':
    (x_train,y_train),(x_test,y_test)=datamanager.load_data()
    
    
    with tf.Session() as sess:
        for filename in x_train:
            image_contents=tf.read_file(filename)
            image=tf.image.decode_jpeg(image_contents,channels=3)
            sess.run(tf.global_variables_initializer())
            img=sess.run((image))
            print(img.shape)
        
    
    #print('x_train size:{}'.format(len(x_train)))
    #print('x_test size:{}'.format(len(x_test)))
   