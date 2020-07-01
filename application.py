import tensorflow as tf
from tensorflow import keras
import numpy as np
from verify import *
from utils import *
import os
from keras import backend as K
import cv2

data=get_existing_data()
# use add_data function to add more data

cap=cv2.VideoCapture('frank_video.mp4')
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface.xml')
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_image=frame[y:y+h,x:x+w]
        cv2.imwrite('frank.jpg',roi_image)
        ans=verification(roi_image,data)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,ans,(x,y),font,1,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow('pic',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
        