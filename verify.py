import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
from keras import backend as K
from utils import *
import cv2

model=keras.models.load_model('facenet_model.h5')
model.load_weights('facenet_weights.h5')


database={}
im=cv2.imread('frank.jpg',cv2.IMREAD_COLOR)
im1=cv2.imread('drogba.jpg',cv2.IMREAD_COLOR)
im2=cv2.imread('drogba2.jpg',cv2.IMREAD_COLOR)
im3=cv2.imread('frank2.jpg',cv2.IMREAD_COLOR)
database['lampard']=img_encoding(im,model)
database['drogba']=img_encoding(im1,model)
database['lampard1']=img_encoding(im2,model)
database['drogba1']=img_encoding(im3,model)


def get_existing_data():
    return database


def add_data(key,img_path):
    img=cv2.imread(img_path,cv2.IMREAD_COLOR)
    database[key]=img_encoding(img,model)
    return database

def verification(image,database):
    encoding=img_encoding(image,model)
    min_dist=100
    for key,value in database.items():
        dist=np.linalg.norm(encoding-value) 
        if dist<min_dist:
            min_dist=dist
            person=key
            
    if min_dist<0.7:
        return person
    else:
        return ' '   