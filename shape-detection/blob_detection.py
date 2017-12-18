#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 11:03:03 2017

@author: ruhshan
"""

# Standard imports
import cv2
import numpy as np;
 
# Read image
im = cv2.imread("dummybgwc.jpg", cv2.IMREAD_GRAYSCALE)
im = cv2.resize(im, (0,0), fx=0.5, fy=0.5) 

edge = cv2.Canny(im, 100, 200)

#gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(edge, (5, 5), 2)
x, thresh = cv2.threshold(edge,0,255, cv2.ADAPTIVE_THRESH_MEAN_C)
# Set up the detector with default parameters.
params = cv2.SimpleBlobDetector_Params()

#params.filterByArea = True
#params.minArea = 2700

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.98
detector = cv2.SimpleBlobDetector_create(params)

# 
## Detect blobs.
keypoints = detector.detect(thresh)
print(keypoints) 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)

cv2.imshow('frame', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
