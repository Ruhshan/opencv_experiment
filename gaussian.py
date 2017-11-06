#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 17:37:12 2017

@author: ruhshan
"""

import cv2
import numpy as np

#img = cv2.imread('bookpage.jpg')
#
#
#grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
#th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
#
##cv2.imshow('original',img)
#cv2.imshow('th', th)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    grayscaled = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    cv2.imshow('frame',frame)
    cv2.imshow('gray', grayscaled)
    cv2.imshow('th', th)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()