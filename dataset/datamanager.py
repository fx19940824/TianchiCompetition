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
        '正常1':0,
        '正常2':1,
        '正常3':2,
        '不导电':3,
        '擦花':4,
        '横条压凹':5,
        '桔皮':6,
        '漏底':7,
        '碰伤':8,
        '起坑':9,
        '凸粉':10,
        '涂层开裂':11,
        '脏点':12,
        '其他':13
    }.get(label_name)

def label2result(label_num):
    return {
        0:'norm',
        1:'norm',
        2:'norm',
        3:'defect1',
        4:'defect2',
        6:'defect4',
        7:'defect5',
        5:'defect3',
        8:'defect6',
        9:'defect7',
        10:'defect8',
        11:'defect9',
        12:'defect10',
        13:'defect11'
    }.get(label_num)

def isDetected(label_num):
    if label_num < 3:
        return 0
    return 1
    

def getfileinDir(data_dir,suffix='/*.jpg'):
    return glob.glob(data_dir+suffix)

def load_traindata(read=False):
    data_dir=os.path.split(os.path.realpath(__file__))[0]
    train_path=os.path.join(data_dir,'traindata/')
    
    x_train=[]
    y_train=[]

    #train_positive_dir=os.path.join(train_path,'无瑕疵样本/')
    #train_negative_dir=os.path.join(train_path,'瑕疵样本/')
    
    train_positive_dir=os.path.join(train_path,'positive/')
    train_negative_dir=os.path.join(train_path,'negative/')

    for input_dir in os.listdir(train_positive_dir):
        pathlist=getfileinDir(os.path.join(train_positive_dir,input_dir))
        labellist=[int(input_dir) for i in range(len(pathlist))]
        #labellist=[str2label_simple(input_dir) for i in range(len(pathlist))]
        x_train.extend(pathlist)
        y_train.extend(labellist)

    for input_dir in os.listdir(train_negative_dir):
        pathlist=getfileinDir(os.path.join(train_negative_dir,input_dir))
        labellist=[int(input_dir) for i in range(len(pathlist))]
        #labellist=[str2label_simple(input_dir) for i in range(len(pathlist))]
        x_train.extend(pathlist)
        y_train.extend(labellist)
    
    if read==True:
        x_data=[]
        for path in x_train:
            x_data.append(read_image(path))
        x_train=x_data

    return np.array(x_train),np.array(y_train)


def load_testdata(read=False):
    data_dir=os.path.split(os.path.realpath(__file__))[0]
    test_path=os.path.join(data_dir,'guangdong_round1_test_a_20180916/')
    label_path=os.path.join(data_dir,'guangdong_round1_submit_sample_20180916.csv')

    x_test=[]
    y_test=[]

    x_test=getfileinDir(test_path)

    with open(label_path,newline='') as csvfile:
        spamreader=csv.reader(csvfile,delimiter=' ',quotechar='|')
        for row in spamreader:
            row=row[0].split(',')
            if row[1]=='defect1':
                y_test.append(1)
            else:
                y_test.append(0)
    
    if read==True:
        x_data=[]
        for path in x_test:
            x_data.append(read_image(path))
        x_test=x_data

    return x_test,y_test

def load_data():
    x_train,y_train=load_traindata()
    x_test,y_test=load_testdata()
            
    return (x_train,y_train),(x_test,y_test)   

def write_result(label,filename=None):
    data_dir=os.path.split(os.path.realpath(__file__))[0]
    label_path=os.path.join(data_dir,'result.csv')

    result=[]
    for i in range(len(label)):
        rowcontent=[]
        if filename==None:
            rowcontent.append(str(i)+'.jpg')
        else:
            rowcontent.append(filename[i])
        rowcontent.append(label2result(label[i]))
        result.append(rowcontent)

    with open(label_path,'w',newline='') as f:
        writer=csv.writer(f)
        writer.writerows(result)


def train_Generator(input_shape,num_classes,batch_size):
    '''x_train,y_train=load_traindata(read=True)

    train_gen=ImageDataGenerator(
        featurewise_center=0,
        featurewise_std_normalization=True,
        rescale=1./255,   
        horizontal_flip=True
    )
    return train_gen.flow(x_train,y_train,batch_size=batch_size)'''

    
    x_train,y_train=load_traindata()

    index=np.arange(x_train.shape[0])
    np.random.shuffle(index)
    x_train,y_train=x_train[index],y_train[index]
    y_train=to_categorical(y_train,num_classes)
    zipped=itertools.cycle(itertools.zip_longest(x_train,y_train))

    while True:
        X=[]
        Y=[]
        for _ in range(batch_size):
            x_path,y=zipped.__next__()
            img_train=img_to_array(load_img(x_path,target_size=(input_shape[0],input_shape[1]))).astype('float32')
            #img_train=ndimage.zoom(img_train,(0.5,0.5,1))
            X.append(normalize(img_train))
            Y.append(y)

        yield np.array(X),np.array(Y)  

def validation_Generator(input_shape,num_classes,batch_size):
    '''x_train,y_train=load_traindata(read=True)

    validation_gen=ImageDataGenerator(
        featurewise_center=0,
        featurewise_std_normalization=True,
        rescale=1./255
    )

    return validation_gen.flow(x_train,y_train,batch_size=batch_size)'''

    x_train,y_train=load_traindata()

    index=np.arange(x_train.shape[0])
    np.random.shuffle(index)
    x_train,y_train=x_train[index],y_train[index]
    y_train=to_categorical(y_train,num_classes)
    zipped=itertools.cycle(itertools.zip_longest(x_train,y_train))

    while True:
        X=[]
        Y=[]
        for _ in range(batch_size):
            x_path,y=zipped.__next__()
            img_train=img_to_array(load_img(x_path,target_size=(input_shape[0],input_shape[1]))).astype('float32')
            X.append(normalize(img_train))
            Y.append(y)

        yield np.array(X),np.array(Y)


def test_Generator(input_shape):
    data_dir=os.path.split(os.path.realpath(__file__))[0]

    x_test=getfileinDir(os.path.join(data_dir,'guangdong_round1_test_a_20180916/'))

    gen=ImageDataGenerator(
        featurewise_center=0,
        featurewise_std_normalization=True,
        data_format=input_shape
    )

    gen.flow_from_directory(x_test,)

    zipped=itertools.cycle(itertools.zip_longest(x_test,y_test))

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

if __name__=='__main__':
    #(x_train,y_train),(x_test,y_test)=load_data()
    test_label=np.random.randint(low=0,high=13,size=300)
    write_result(test_label)

    