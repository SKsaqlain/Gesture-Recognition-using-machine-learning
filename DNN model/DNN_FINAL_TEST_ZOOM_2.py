#importing the required libraries

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import time



#loading the mnist data set.
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder("float", shape=[None, 784])

y_ = tf.placeholder("float", shape=[None, 10])


#network parameters
L = 2000
M = 1000
N = 600
O = 300

W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))  # 784 = 28 * 28
B1 = tf.Variable(tf.zeros([L]))
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.zeros([M]))
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3 = tf.Variable(tf.zeros([N]))
W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B4 = tf.Variable(tf.zeros([O]))
W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))


#using relu activation function and softwax at the last layer.
Y1 = tf.nn.relu(tf.matmul(x, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
y= tf.nn.softmax(Ylogits)




cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100
learning_rate = 0.003
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)




init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
hm_epochs=10
batch_size=100
for k in range(hm_epochs):
    for i in range(int(mnist.train.num_examples/batch_size)):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print("epoch {} completed".format(k+1))
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print ("accuracy={} %".format(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})*100))
    
print('*************************training done*************************')


#loading the image to be zoomed.

#gesture 0->zoom into 1st quadrent
#gesture 1->zoom into 2nd quadrent
#gesture 2->zoom into 3rd quadrent
#gesture 3->zoom into 4th quadrent
#gesture 4->go back to the previous image.

#once the gesture is done press 'q' once to zoom into that region
#press 'n' to clear the drawing area.
#press 'q' multiple times to quit.

test_image=cv2.imread("./colorFull.jpg",1) 
test_image_1=cv2.resize(test_image,(640,480),interpolation=cv2.INTER_CUBIC)    
cv2.imshow("test_image",test_image_1)
image_list=[test_image_1]

def goto(test_image,image_list):
    cap = cv2.VideoCapture(1)
    
	    
    def smooth(l):
        sum_x=0
        sum_y=0
        length=len(l)
        if len(l)<3:
            return (l[-1])
        else:
            for i in range(3):
                sum_x+=l[length-1-i][0]
                sum_y+=l[length-1-i][1]
        avg=(sum_x//3,sum_y//3)
        return avg
            
        
        
    count=0
    time0=time.time()
    #_,mask_frame=cap.read()
    #begin our 'infinite' while loop
    
    
    face_cascade = cv2.CascadeClassifier('../XML/haarcascade_frontalface_default.xml')
    
    fist_cascade=cv2.CascadeClassifier("../XML/hand.xml")
    hand_cascade=cv2.CascadeClassifier("../XML/palm.xml")
    paint=np.zeros((600,600 ,3),np.uint8)
    paint[:]=[255,255,255]
    l=list()
    while(1):
        
        ret,frame=cap.read()
        img=frame
        img=cv2.flip(img,1)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fist = fist_cascade.detectMultiScale(gray, 1.3, 5)
        hand=hand_cascade.detectMultiScale(gray,1.3,5)
        
        #hand=hand_cascade
        
    	#getting the co-ordinate of the fist which acts like a air-brush
        for (xb,yb,w,h) in fist:
            frame=cv2.rectangle(frame,(xb,yb),(xb+w,yb+h),(255,0,0),3)
            cx=640-int(xb+w/2)
            cy=int(yb+h/2)
            cx = int((cx / 640 )* 600)
            cy = int((cy / 480 )*600)
    
            l=l+[(cx,cy)]
            co_ord=smooth(l)
            #drawing a small circle at the centroid of the fist.
            paint=cv2.circle(paint,co_ord,5,(0,0,0),-1)
        img=frame
        img=cv2.flip(img,1)
        cv2.imshow("frame",img)
    
        
        cv2.imshow("paint",paint)
        
        
        k=cv2.waitKey(1)
        if k==ord("q"):
            break
        elif k==ord("n"):
            paint[:]=[255,255,255]
            l=[]
    
    cap.release()
    cv2.destroyAllWindows()
    
    blah = np.zeros((600,600,3), np.uint8)
    #blah[:] =[255,255,255]
    for i in range(1,len(l)):
        blah = cv2.line(blah, (l[i-1][0],l[i-1][1]),(l[i][0],l[i][1]),(255,255,255),10)   
    
    #res = cv2.resize(blah,(20,20), interpolation = cv2.INTER_CUBIC)
    
    
    #cv2.imwrite('image.jpg',blah)   
    
    cv2.destroyAllWindows()
    
    
    
    
    imgray = cv2.cvtColor(blah,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    img = cv2.drawContours(thresh, contours, -1, (0,255,0), 3)
    
    cnt = contours[0]
    M = cv2.moments(cnt)
    
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    print(cx,cy)
    
    xb,yb,w,h = cv2.boundingRect(cnt)
    thresh = cv2.rectangle(thresh,(xb,yb),(xb+w,yb+h),(0,255,0),2)
    
    
    
    res = blah[yb:yb+h,cx-h//2:cx+h//2]
    
    res = cv2.resize(res,(20,20), interpolation = cv2.INTER_CUBIC)
    
    
    pad = np.zeros((28,28,3), np.uint8)
    pad[4:24,4:24] = res
    
    cv2.imwrite('./kill.jpg',pad)
    
    
    imgray = cv2.cvtColor(pad,cv2.COLOR_BGR2GRAY)
    img=cv2.imread("./kill.jpg",0)
    
    img=[np.ravel(img)]
    #print(img1)
    
    prediction=tf.argmax(y,1)
    print('hello')
    pred=prediction.eval(feed_dict={x: img}, session=sess)
    print ("predictions", pred)
    
    #function to zoom into a specific area.
    def zoom_img(x,image,dimension):
        global image_list
        dimension=image.shape
        row=dimension[0]
        half_row=int(row/2)
        
        col=dimension[1]
        half_col=int(col/2)
        
        if x==0:
            zoom=cv2.resize(image[0:half_row,half_col:col],(640,480),interpolation=cv2.INTER_CUBIC)
            print("0")
            return(zoom)
        elif x==1:
            zoom=cv2.resize(image[0:half_row,0:half_col],(640,480),interpolation=cv2.INTER_CUBIC)
            print(1)
            return (zoom)
        elif x==2:
            zoom=cv2.resize(image[half_row:row,0:half_col],(640,480),interpolation=cv2.INTER_CUBIC)
            print(2)
            return (zoom)
        elif x==3:
            zoom=cv2.resize(image[half_row:row,half_col:col],(640,480),interpolation=cv2.INTER_CUBIC)
            print(3)
            return (zoom)
        elif x==4:
            zoom=image_list[0]
            return (zoom)
        
            
        else:
            return None
            
    
    
    dimension=test_image.shape
    cv2.imshow("image",test_image_1)

    zoom=zoom_img(pred[0],test_image,dimension)
    image_list.append(zoom)
    
    try:
        cv2.imshow("zoomed_image",zoom)
        goto(zoom,image_list)
    except:
        return None
   
testing=goto(test_image,image_list) 
if testing ==None:

    cv2.destroyAllWindows()