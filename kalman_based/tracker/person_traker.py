from centroidtracker import CentroidTracker
from imutils.video import VideoStream
import imutils

import time
from PIL import Image


import os

import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import struct
import cv2
from numpy import expand_dims
import tensorflow as tf
from skimage.transform import resize
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.models import load_model, Model
from keras.layers.merge import add, concatenate
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.patches import Rectangle
from tracker import Tracker
import time
    


# In[22]:


net_h,net_w = 416,416

obj_thresh ,nms_thresh = 0.5,0.45

anchors = [[116,90 ,156,198 ,373,326],[30,61, 62,45 ,59,119],
           [10,13,16, 33,23]]

labels = ["person"]

# In[3]:


net_h,net_w = 416,416

obj_thresh ,nms_thresh = 0.5,0.45

anchors = [[116,90 ,156,198 ,373,326],[30,61, 62,45 ,59,119],
           [10,13,16, 33,23]]

labels = ["person"]


# In[4]:



# In[3]:



# In[5]:


yolov3 = load_model('yolov3.h5')


# In[6]:


from numpy import expand_dims
import cv2


# In[7]:


def load_image_pixels(filename , shape):
    #load the image to get its shape
    image = load_img(filename)
#     image = filename
    width,height = image.size
    
    
    #load the image with the required size
    image = load_img(filename,target_size=shape)
    
    #convert the image to numpy array
    
    image = img_to_array(image)
    
    image = image.astype('float32')
    image/=255.0
    
    
    image = expand_dims(image,0)
    return image,width,height
    
    


# In[8]:


input_w,input_h=416,416

# photo_filename = 'meme.jpg'


# image , image_w,image_h = load_image_pixels(photo_filename , (input_w,input_h))


# In[9]:


class BoundBox:
    def __init__(self,xmin,ymin,xmax,ymax,objness=None,classes=None):
        
        
        self.xmin=xmin
        self.xmax=xmax
        self.ymin=ymin
        self.ymax=ymax
        
        
        self.objness = objness
        
        self.classes = classes
        
        self.label = -1
        self.score = -1
        
        
        def get_label(self):
            if self.label == -1:
                self.label = np.argmax(self.classes)
            
            return self.label
        
        
        def get_score(self):
            if self.score == -1:
                self.score = self.classes[self.get_label()]
            return self.score
        
def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3 

        
        
def bbox_iou(box1,box2):
    intersect_w = _interval_overlap([box1.xmin,box1.xmax],[box2.xmin,box2.xmax])
    
    intersect_h = _interval_overlap([box1.ymin,box1.ymax],[box2.ymin,box2.ymax])
    
    intersection = intersect_w * intersect_h
    
    height_box1 = abs(box1.ymax - box1.ymin)
    width_box1 = abs(box1.xmin - box1.xmax)
    height_box2 = abs(box1.ymax - box1.ymin)
    width_box2 = abs(box2.xmax - box2.xmin)
    
    area1 = height_box1 * width_box1
    area2 = height_box2 * width_box2
    
    union = area1 + area2 - intersection
    
    return float(intersection) / float(union)


def do_nms(boxes , nms_thresh):
    if len(boxes) > 0:
        nb_classes = len(boxes[0].classes)
    else:
        return
    
    
    for c in range(nb_classes):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].classes[c] == 0: continue
                
            for j in range(i+1 , len(sorted_indices)):
                index_j = sorted_indices[j]
                
                if bbox_iou(boxes[index_i],boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c]=0
            


# In[10]:


def decode_netout(netout , anchors , obj_thres , net_h , net_w):
    
    grid_h , grid_w = netout.shape[:2]
    
    nb_box = 3
    
    netout = netout.reshape((grid_h,grid_w,nb_box,-1))
    
    nb_classes = netout.shape[-1] - 5
    
    boxes = []
    
    netout[...,:2] = _sigmoid(netout[...,:2])
    
    netout[... , 4:] = _sigmoid(netout[...,4:])
    
    netout[... , 5:] = netout[...,4][...,np.newaxis]*netout[...,5:]
    
    netout[...,5:] *= netout[...,5:] > obj_thresh
    
    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                
                
                #forth row is the objectness score
                objectness = netout[int(row)][int(col)][b][4]
                
                
                if objectness <= obj_thres: continue
                    
                    
                x,y,w,h = netout[int(row)][int(col)][b][:4]
                
                
                x = (col + x)/ grid_w
                y = (row + y)/grid_h
                w = (anchors[2*b + 0] * np.exp(w)) / net_w
                h = (anchors[2*b  + 1] * np.exp(h)) / net_h
                
                
                classes = netout[int(row)][int(col)][b][5:]
                
                box = BoundBox(x - w/2, y - h/2, x + w/2,y + h/2,objectness,classes)
                
                boxes.append(box)
                
    return boxes    
                


# In[11]:


#bounding boxes will be stretched back into the shape of the original image
#will allow plotting the original image and draw the bounding boxes, hopefully detecting real objects.
# correct the sizes of the bounding boxes for the shape of the image
#correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
   if (float(net_w)/image_w) < (float(net_h)/image_h):
       new_w = net_w
       new_h = (image_h*net_w)/image_w
   else:
       new_h = net_w
       new_w = (image_w*net_h)/image_h
       
   for i in range(len(boxes)):
       x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
       y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
       
       boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
       boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
       boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
       boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


# In[12]:


# suppress non-maximal boxes
#do_nms(boxes, 0.5)


# In[13]:








# get all of the results above a threshold
# takes the list of boxes, known labels, 
#and our classification threshold as arguments and returns parallel lists of boxes, labels, and scores.
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
                # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores




# In[16]:


# v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)


# In[17]:


anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119],[10,13, 16,30, 33,23]]
class_threshold = 0.6
from PIL import Image

ct = CentroidTracker()

cap = cv2.VideoCapture('ff.mp4')

tracker = Tracker(150,30,5)

while True:
    
    ret,frame = cap.read()
    # cap.read()
    # cap.read()
    # cap.read()
    frame = cv2.resize(frame,(800,800))
    image = Image.fromarray(frame,'RGB')

    image.save('myimg.jpeg')

    image,image_w,image_h = load_image_pixels('myimg.jpeg',(net_w,net_w))

    yolos = yolov3.predict(image)    
    boxes = list()

    for i in range(len(yolos)):
        boxes+=decode_netout(yolos[i][0],anchors[i],obj_thresh,net_h,net_w)

    correct_yolo_boxes(boxes,image_h,image_w,net_h,net_w)

    do_nms(boxes,nms_thresh)

    v_boxes,v_labels,v_scores=get_boxes(boxes,labels,class_threshold)

    centers=[]
    rects=[]

    for i in range(len(v_boxes)):
        box = v_boxes[i]

        box_cords=(x1,y1,x2,y2)= (box.xmin,box.ymin,box.xmax,box.ymax)
        c_x=(x1+x2)/2
        c_y=(y1+y2)/2
        rects.append(box_cords)
        centers.append((c_x,c_y))
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
    centers = np.array(centers)
    if(len(rects)>0):
        tracker.update(centers)
        for j in range(len(tracker.tracks)):
            if (len(tracker.tracks[j].trace) > 1):
                x=int(tracker.tracks[j].trace[-1][0,0])
                y=int(tracker.tracks[j].trace[-1][0,1])
                cv2.putText(frame,str(tracker.tracks[j].trackId), (x,y),0, 0.5,(124,252,0),2)
                for k in range(len(tracker.tracks[j].trace)):
                    x=int(tracker.tracks[j].trace[k][0,0])
                    y=int(tracker.tracks[j].trace[k][0,1])
                    # cv2.circle(frame,(x,y),3,(124,252,0),-1)
                cv2.circle(frame,(x,y),6,(124,252,0),-1)


    cv2.imshow("frame",frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.stop()


