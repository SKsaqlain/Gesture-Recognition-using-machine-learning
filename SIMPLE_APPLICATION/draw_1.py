
#importing the necessary packages
import cv2
import time

mem_x=0
mem_y=0
mem_w=10
mem_h=10

cap = cv2.VideoCapture(1)



count=0
time0=time.time()
#_,mask_frame=cap.read()
#begin our 'infinite' while loop

flag = 0
#face_cascade = cv2.CascadeClassifier('C:/Users/Saqlain/Desktop/haarcascade_frontalface_default.xml')
fist_cascade=cv2.CascadeClassifier("../XML/hand.xml")#path to the hand.xml file 
hand_cascade=cv2.CascadeClassifier("../XML/palm.xml")#path to the plam.xml file
while(1):
    try:
        ret,frame=cap.read()
        img=frame
        img1=cv2.flip(img,1)
        im = cv2.imread('./colorFull.jpg',1)#path to any image
        im=cv2.resize(im,(640,480), interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        fist = fist_cascade.detectMultiScale(gray, 1.3, 5)
        hand=hand_cascade.detectMultiScale(gray,1.3,5)
        
        #hand=hand_cascade
        if (len(fist)==1):
            for (x,y,w,h) in fist:
                frame=cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),3)
                im = cv2.rectangle(im,(x,y),(x+w, y+h),(255,0,0),2)
                mem_x = x
                mem_y = y
                mem_w = w
                mem_h = h
                cv2.imshow("zoom",im)
    
    
    
        
        elif len(hand) == 1:
            for (x,y,w,h) in hand:
                frame=cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),3)
            res=cv2.resize(im[mem_y:mem_y+mem_h,mem_x:mem_x+mem_w] ,(640,480), interpolation = cv2.INTER_AREA)
            cv2.imshow("zoom",res)
        
    
        cv2.imshow("frame",img1)    
        k=cv2.waitKey(1)
        if k==ord("q"):
            break
    except :
        pass
    
        
cap.release()
cv2.destroyAllWindows()
        
    