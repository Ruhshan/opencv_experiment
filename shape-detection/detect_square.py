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

image = cv2.imread('dummybgwc.jpg', cv2.IMREAD_COLOR)
image = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
ratio = image.shape[0] / float(image.shape[0])

edge = cv2.Canny(image, 100, 200)



new_thresh =edge
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
print(max_square)

x, y, w, h = cv2.boundingRect(max_square[1])

print(x,y,w,h)
roi=image[y:y+h,x:x+w]
#cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
#            0.5, (255, 255, 100), 2)
#cv2.drawContours(image, max_square[1], -1, (0, 255, 0), 5)
cv2.imshow('frame', image)
cv2.imshow('roi', roi)

cv2.waitKey(0)
cv2.destroyAllWindows()