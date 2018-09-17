import os
import csv
import numpy as np
from PIL import Image
from scipy import ndimage


def read_image(input_path,scale=1):
    data=np.array(Image.open(input_path).convert('RGB'))
    data=ndimage.zoom(data,(scale,scale,1))
    return data
    
def str2label(label_name):
    return {
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

def load_data():
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
    
    for image_path in os.listdir(train_positive_dir):
        x_train.append(os.path.join(train_positive_dir,image_path))
        #x_train.append(read_image(input_path=os.path.join(train_positive_dir,image_path),
        #                        scale=0.1))
        y_train.append('正常')
    for input_dir in os.listdir(train_negative_dir):
        folder=os.path.join(train_negative_dir,input_dir)
        for image_path in os.listdir(folder):
            x_train.append(os.path.join(folder,image_path))
            #x_train.append(read_image(input_path=os.path.join(folder,image_path),
            #                        scale=0.1))
            y_train.append(input_dir)
    
    for image_path in os.listdir(test_path):
        x_test.append(os.path.join(test_path,image_path))
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

if __name__=='__main__':
    (x_train,y_train),(x_test,y_test)=load_data()

    