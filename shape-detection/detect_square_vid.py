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
while True:
    #image = cv2.imread('dummybgwc.jpg', cv2.IMREAD_COLOR)
    ret, image = cap.read()
    #image = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
    ratio = image.shape[0] / float(image.shape[0])
    mask = np.zeros((480, 640, 1), dtype = "uint8")
    not_mask = cv2.bitwise_not(mask)
    cm = cv2.circle(mask,(320, 240), 220, (255,255,255),-1)
    
    
    
    edge = cv2.Canny(image, 100, 200)
    
    output = cv2.bitwise_and(edge, cm)
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
        img_croped = crop_minAreaRect(image, rect)
        cv2.imshow('crop', img_croped)
    except:
        pass
    
#    try:
#        x, y, w, h = cv2.boundingRect(max_square[1])
#        
#        
#        n_area = w * h
#        #print(area, n_area, area-n_area, end="*")
#        if n_area> 50000 and n_area < 90000:
#            area = n_area
#            #print(x,y,w,h)
#            
#            proi=new_thresh[y:y+h,x:x+w]
#            roi = image[y:y+h, x:x+w]
#            #roi = get_wighted(roi)
#            im = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#            
#            
#            #cv2.imshow('im', im)
##            blurred = cv2.GaussianBlur(im, (5, 5), 2)
##            cv2.imshow("blur", blurred)
#            #xi, threshi = cv2.threshold(blurred,0,255, cv2.ADAPTIVE_THRESH_MEAN_C)
#            #cv2.imshow('thresh', threshi)
#            edgei = cv2.Canny(im, 200, 250,5)
#            cnts_i = cv2.findContours(edgei.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#            cnts_i = cnts_i[0] if imutils.is_cv2() else cnts_i[1]
#            #cv2.imshow('edgei', edgei)
#            ratioi = roi.shape[0] / float(roi.shape[0])
#            cv2.imshow('roi', roi)
#            cv2.imshow("cropped", img_croped)
#            
##            for ci in cnts_i:
##                     # compute the center of the contour, then detect the name of the
##                     # shape using only the contour
##                M = cv2.moments(ci)
##            
##                try:
##                    cXi = int((M["m10"] / M["m00"]) * ratio)
##                    cYi = int((M["m01"] / M["m00"]) * ratio)
##                except:
##                    cXi = 400
##                    cYi = 500
##            
##                shapei, peri = sd.detect(ci)
##                shape_array.append(shapei)
##            
##                 #
##                 # # multiply the contour (x, y)-coordinates by the resize ratio,
##                 # # then draw the contouqrs and the name of the shape on the image
##                ci = ci.astype("float")
##                ci *= ratioi
##                ci = ci.astype("int")
##                
##                cv2.drawContours(roi, [ci], -1, (0, 255, 0), 1)
##                cv2.putText(roi, shapei, (cXi, cYi), cv2.FONT_HERSHEY_SIMPLEX,
##                    0.5, (255, 255, 100), 2)
##                
##                cv2.imshow('roi', roi)
##                #cv2.imshow('nroi', roi)
#            
#            
#            
#    except:
#        pass
    #cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
    #            0.5, (255, 255, 100), 2)
    cv2.circle(image,(320, 240), 220, (78,120,99), 3)
    cv2.drawContours(image, max_square[1], -1, (0, 255, 0), 5)
    cv2.imshow('frame', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()