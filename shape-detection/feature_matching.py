#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 11:20:12 2017

@author: ruhshan
"""

#import numpy as np
import cv2
import numpy as np
import argparse
import imutils

#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#	help="path to the input image")
#args = vars(ap.parse_args())

#tmplt = cv2.imread(args["image"])
template = cv2.imread('marker_2r.png', 0)
w, h = template.shape[::-1]
#tmplt = cv2.imread('watch_2.jpg', cv2.IMREAD_GRAYSCALE)
#rint(tmplt)

#cv2.imshow('t', tmplt)
#cv2.imshow('image', img)

cap = cv2.VideoCapture(0)
cv2.namedWindow('extract', cv2.WINDOW_FULLSCREEN)
while True:
    ret, img_rgb =  cap.read()
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray,template,cv2.CV_TM_CCORR_NORMED)
    
    threshold = 0.7
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        print(pt)
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        extract = img_rgb[pt[1]:pt[1]+h, pt[0]:pt[0]+w]
        
        cv2.imshow('extract', extract)
    
    cv2.imshow('frame', img_rgb)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
