#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:27:48 2017

@author: ruhshan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 12:30:17 2017

@author: ruhshan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 09:45:46 2017

@author: ruhshan
"""

from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
import numpy as np
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#     help="path to the input image")
# args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
cap = cv2.VideoCapture(0)
area=1
template_h =  cv2.imread('hor_connect.png',0)
tw, th = template_h.shape[::-1]
def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
#    if angle < -45:
#        angle+=90
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], 
                       pts[1][0]:pts[2][0]]

    return img_crop
def rm_logo(img):
    gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w, c = img.shape
    print(h, w)
    print("tt",tw, th)
    #imgu = img[:round(h*0.4)+5,]
    #imgd = img[round(h*0.6)+8:,]
    #cv2.imshow('rmu', imgu)
    #cv2.imshow('rmd', imgd)
    res = cv2.matchTemplate(gr,template_h,cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + tw, top_left[1] + th)
    cv2.rectangle(img,top_left, bottom_right, 255, 2)
    #print("res",res)
    #connectless = np.concatenate((imgu, imgd))
    cv2.imshow("conless", gr)
    
while True:
    #image = cv2.imread('dummybgwc.jpg', cv2.IMREAD_COLOR)
    ret, image = cap.read()
    #image = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
    ratio = image.shape[0] / float(image.shape[0])
    mask = np.zeros((480, 640, 1), dtype = "uint8")
    not_mask = cv2.bitwise_not(mask)
    cm = cv2.circle(mask,(320, 240), 220, (255,255,255),-1)
    
    
    
    edge = cv2.Canny(image, 100, 200)
    gr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #edge = cv2.Laplacian(gr, cv2.CV_8UC4)
    output = cv2.bitwise_and(edge, cm)
    #cv2.imshow("edge", edge)
    #cv2.imshow("masked", output)
    
    new_thresh = output
        # find contours in the thresholded image and initialize the
        # shape detector
    cnts = cv2.findContours(new_thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    sd = ShapeDetector()
        
    #cv2.imshow('thresh',new_thresh)
    shape_array=[]
    squares = []
    for c in cnts:
             # compute the center of the contour, then detect the name of the
             # shape using only the contour
        M = cv2.moments(c)
    
        try:
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
        except:
            cX = 400
            cY = 500
    
        shape, peri = sd.detect(c)
        shape_array.append(shape)
    
         #
         # # multiply the contour (x, y)-coordinates by the resize ratio,
         # # then draw the contouqrs and the name of the shape on the image
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        if shape=="s":
            squares.append([peri, c])
    #        cv2.drawContours(image, [c], -1, (0, 255, 0), 1)
    #        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
    #            0.5, (255, 255, 100), 2)
    max_square = [0, []]
    for s in squares:
        if s[0]>max_square[0]:
            max_square = s 
    #print(max_square)
    try:
        rect = cv2.minAreaRect(max_square[1])
        w, h = rect[1]
        n_area = w * h
        if n_area > 50000 and n_area < 80000:
            img_croped = crop_minAreaRect(image, rect)
            #cv2.imshow('auto', img_croped)
            rm_logo(img_croped)
            area = n_area
    except:
        pass
    
    #cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
    #            0.5, (255, 255, 100), 2)
    cv2.circle(image,(320, 240), 220, (78,120,99), 3)
    cv2.drawContours(image, max_square[1], -1, (0, 255, 0), 5)
    cv2.imshow('frame', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()