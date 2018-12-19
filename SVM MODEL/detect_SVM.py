

mem_x=0
mem_y=0
#importing various modules
import cv2
import numpy as np
import sys
import os
import time 
from sklearn import svm

filenames=os.listdir("../DATASETS/datasets_re")#path to the dataset of hands

 
#reading dataset for training

X=[]
Y=[]
for i in filenames:
    img = cv2.imread("../DATASETS/datasets_re/"+i,0)
    a = np.ravel(img)#covetring matrix into a vector
    Y.append(int(i[3]))
    X.append(a)

x =np.array(X)
print(len(X),len(Y))
y=np.array(Y)
print(y)
c=1
h=0.02
clf = svm.SVC(kernel="linear",gamma=0.7,C=c)#linear svm model
print(clf.fit(x,y))

cap = cv2.VideoCapture(1)


count=0
time0=time.time()
#using haarcascade to eliminate face from the frame is present.
face_cascade = cv2.CascadeClassifier('../XML/haarcascade_frontalface_default.xml')

while(cap.isOpened()):
    #read the streamed frames (we previously named this cap)
    
    _,frame=cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        frame[y:y+h, x:x+w]=0
        
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        frame[y:y+h, x:x+w]=0
    #it is common to apply a blur to the frame
    frame=cv2.GaussianBlur(frame,(5,5),0)
    
   # convert from a BGR stream to an HSV stream
   #using the HSV value of the hand to detect skin color from the frame.
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    HSVLOW=np.array([0,20,0])
    HSVHIGH=np.array([40,244,255])
    mask = cv2.inRange(hsv,HSVLOW, HSVHIGH)
    res = cv2.bitwise_and(frame,frame, mask = mask)

    
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    (thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   


    
    ret, thresh = cv2.threshold(im_bw, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img = cv2.drawContours(im_bw, contours, -1, (0,255,0), 3)

    #calculating the  area of the contours and finding the maximum area
    max_area = 0.0
    pnt = 0
    i = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>max_area:
            pnt = i
            max_area = area
        i+=1
    
    #taking the largest contour
    cnt =contours[pnt]
    x,y,w,h = cv2.boundingRect(cnt)
    frame = cv2.rectangle(im_bw,(x,y),(x+w,y+h),(255,0,0),5)


    cx = w+(w/2) 
    cy = y+(h/2)
    diff_x = cx - mem_x
    diff_y = cy - mem_y
    font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(frame,str(count),(100,100), font, 2,(255,255,255),2,cv2.LINE_AA)

    cv2.imshow('frame',im_bw)
    
    
    #NUMBER=number(im_bw[y:y+h,x:x+w])
    #print(NUMBER)
    
    im_bw=im_bw[y:y+h,x:x+w]
    res=cv2.resize(im_bw,(25,25),interpolation=cv2.INTER_AREA)
    a_test = np.ravel(res)
    
    result=int(clf.predict([a_test]))
    print(result)
    
    frame = cv2.flip(frame,180)
    cv2.putText(frame,str(result),(100,100), font, 2,(255,255,255),2,cv2.LINE_AA)
    
    cv2.imshow('rect',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count+=1
    
    
    
    
    
    
print(count)
time=time.time()-time0
print(time) 
cap.release()
cv2.destroyAllWindows()