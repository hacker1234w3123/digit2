import cv2
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time

if (not os . environ.get("PYTHONHTTPSVERIFY",'') and getattr(ssl,'_create_unverified_context',None )):
    ssl._create_default_https_context = ssl._create_unverified_context

X,y = fetch_openml('mnist_784',version = 1,return_X_y = True)
print(len(X))
print(pd.Series(y).value_counts())
X = np.array(X)
classes = ['0', '1', '2','3', '4','5', '6', '7', '8', '9']
nClasses = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
model = LogisticRegression(solver = 'saga',multi_class= "multinomial")
model.fit(X_train_scaled , y_train)
yPredict = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test,yPredict)
print(accuracy)

cam = cv2.videoCapture(0)

while(True):
    try:
         ret,frame = cam.read()
         grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
         height,width = grey.shape
         upper_left = (int(width/2-56),int(height/2-56))
         bottom_height = (int(width/2+56),int(height/2+56))
         cv2.rectangle(grey,upper_left,bottom_height,(0,255,0),2)
         roi = grey[upper_left[1]:bottom_height[1],upper_left[0]:bottom_height[0]]
         iampil = Image.fromarray(roi)
         image_bw = iampil.convert('L')
         image_bw_resized = image_bw.resize((28,28),Image.ANTIALIAS)
         image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized) 
         pixel_filter = 20
         min_pixel = np.percentile(image_bw_resized_inverted,pixel_filter)
         image_bw_resized_inverted_sacled = np.clip(image_bw_resized_inverted - min_pixel , 0,255)
         max_pixel = np.max(image_bw_resized_inverted)
         image_bw_resized_inverted_sacled = np.asarray(image_bw_resized_inverted_sacled) / max_pixel
         testSample = np.array (image_bw_resized_inverted_sacled).reshape(1,784)
         y_Predict = model.predict(testSample)
         print("the predicted value os this " , y_Predict)
         cv2.imshow("farme",grey)
    except Exception as e :
       pass
cam.release()
cv2.destroyAllWindows()

