# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2

img = cv2.imread('watch.jpg', cv2.IMREAD_COLOR)

cv2.line(img, (0,0), (500,500), (0,0,0),30)

cv2.rectangle(img, (0,0), (500,500), (0,0,0),30)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()