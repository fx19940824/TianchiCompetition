import os
import glob
import csv
import numpy as np
from PIL import Image
from scipy import ndimage
import itertools
from keras.preprocessing.image import img_to_array,load_img,ImageDataGenerator
from keras.utils import normalize,to_categorical

def get_traindir():
    data_dir=os.path.split(os.path.realpath(__file__))[0]
    train_path=os.path.join(data_dir,'guangdong_round1_train2_20180916/')
    return train_path

def get_testdir():
    data_dir=os.path.split(os.path.realpath(__file__))[0]
    test_path=os.path.join(data_dir,'guangdong_round1_test_a_20180916/')
    return test_path

def read_image(input_path,scale=1):
    data=np.array(Image.open(input_path).convert('RGB'))
    data=ndimage.zoom(data,(scale,scale,1))
    return data
    
def str2label_complex(label_name):
    return {
        '正常':0,
        '变形':1,
        '驳口':2,
        '擦花':3,
        '打白点':4,
        '杂色':5,
        '打磨印':6,
        '返底':7,
        '漏底':8,
        '不导电':9,
        '挂具印':10,
        '碰凹':11,
        '划伤':12,
        '火山口':13,
        '桔皮':14,
        '铝屑':15,
        '喷流':16,
        '漆泡':17,
        '气泡':18,
        '起坑':19,
        '纹粗':20,
        '涂层开裂':21,
        '脏点':22,
        '粘接':23,
        '凸粉':24,
        '喷涂划伤':25,
        '拖伤':26,
        '油印':27
    }.get(label_name)

def str2label_simple(label_name):
    return {
        '正常':0,
        '不导电':1,
        '擦花':2,
        '横条压凹':3,
        '桔皮':4,
        '漏底':5,
        '碰伤':6,
        '其他':7,
        '起坑':8,
        '凸粉':9,
        '涂层开裂':10,
        '脏点':11
    }.get(label_name)

def getfileinDir(data_dir,suffix='/*.jpg'):
    return glob.glob(data_dir+suffix)

def load_data_simple():
    data_dir=os.path.split(os.path.realpath(__file__))[0]
    
    #train1_path=os.path.join(data_dir,'guangdong_round1_train1_20180903.zip')
    train2_path=os.path.join(data_dir,'guangdong_round1_train2_20180916/')
    test_path=os.path.join(data_dir,'guangdong_round1_test_a_20180916/')
    label_path=os.path.join(data_dir,'guangdong_round1_submit_sample_20180916.csv')

    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]

    train_positive_dir=os.path.join(train2_path,'无瑕疵样本/')
    train_negative_dir=os.path.join(train2_path,'瑕疵样本/')
    
    i=0
    for image_path in os.listdir(train_positive_dir):
        x_train.append(os.path.join(train_positive_dir,image_path))
        y_train.append(0)
        if i==0:
            break
            
    return (x_train,y_train),(x_test,y_test)  

def load_data():
    data_dir=os.path.split(os.path.realpath(__file__))[0]
    
    train_path=os.path.join(data_dir,'guangdong_round1_train2_20180916/')
    test_path=os.path.join(data_dir,'guangdong_round1_test_a_20180916/')
    label_path=os.path.join(data_dir,'guangdong_round1_submit_sample_20180916.csv')

    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]

    train_positive_dir=os.path.join(train_path,'无瑕疵样本/')
    train_negative_dir=os.path.join(train_path,'瑕疵样本/')
    
    for input_dir in os.listdir(train_positive_dir):
        pathlist=getfileinDir(os.path.join(train_positive_dir,input_dir))
        labellist=[str2label_simple(input_dir) for i in range(len(pathlist))]
        x_train.extend(pathlist)
        y_train.extend(labellist)

    for input_dir in os.listdir(train_negative_dir):
        pathlist=getfileinDir(os.path.join(train_negative_dir,input_dir))
        labellist=[str2label_simple(input_dir) for i in range(len(pathlist))]
        x_train.extend(pathlist)
        y_train.extend(labellist)
    
    x_test=getfileinDir(test_path)
    #for image_path in os.listdir(test_path):
        #x_test.append(os.path.join(test_path,image_path))
        #x_test.append(read_image(input_path=os.path.join(test_path,image_path)))

    with open(label_path,newline='') as csvfile:
        spamreader=csv.reader(csvfile,delimiter=' ',quotechar='|')
        for row in spamreader:
            row=row[0].split(',')
            if row[1]=='defect1':
                y_test.append(1)
            else:
                y_test.append(0)
            
    return (x_train,y_train),(x_test,y_test)   

def train_Generator(input_shape,batch_size):
    (x_train,y_train),(x_test,y_test)=load_data()
    y_train=to_categorical(y_train)
    zipped=itertools.cycle(itertools.zip_longest(x_train,y_train))

    while True:
        X=[]
        Y=[]
        for _ in range(batch_size):
            x_path,y=zipped.__next__()
            img_train=img_to_array(load_img(x_path)).astype('float32')
            img_train=ndimage.zoom(img_train,(0.1,0.1,1))
            X.append(normalize(img_train))
            Y.append(y)

        yield np.array(X),np.array(Y)    

def test_Generator():
    (x_train,y_train),(x_test,y_test)=load_data()

    zipped=itertools.cycle(itertools.zip_longest(x_test,y_test))


if __name__=='__main__':
    (x_train,y_train),(x_test,y_test)=load_data()

    