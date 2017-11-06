#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 16:19:50 2017

@author: ruhshan
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2

img = cv2.imread('watch_2.jpg', cv2.IMREAD_COLOR)

x, y, c = img.shape

for i in range(x):
    ct=0
    for j in range(y):
        ct+=1
        if ct%20==0:
            img[i,j]=[ct,255,ct]

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()