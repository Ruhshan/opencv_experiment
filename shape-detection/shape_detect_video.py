# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages
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

while True:
    ret, image =  cap.read()
    #image = cv2.imread(args["image"])
    #resized = imutils.resize(image, width=300)
    edge = cv2.Canny(image, 100, 200)
    #cv2.imshow('edge', edge)
    #image = edge
    ratio = image.shape[0] / float(image.shape[0])
    
    mask = np.zeros((480, 640, 1), dtype = "uint8")
    not_mask = cv2.bitwise_not(mask)
    cm = cv2.circle(mask,(320, 240), 200, (255,255,255),-1)
    
    
    
    
    
    
    gray_i = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output = cv2.bitwise_and(gray_i, cm)
    
    #cv2.imshow('output', output)
    
    

    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_OTSU, 115, 1)
    
    x, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)[1]
    
    new_thresh = cv2.bitwise_and(edge, cm)
    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(new_thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    sd = ShapeDetector()
    
    cv2.imshow('thresh',new_thresh)
    #print(cnts)q

    #loop over the contours
    ct=0
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

        shape = sd.detect(c)
    
         #
         # # multiply the contour (x, y)-coordinates by the resize ratio,
         # # then draw the contouqrs and the name of the shape on the image
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 2)
    
         # show the output image

    cv2.circle(image,(320, 240), 200, (78,120,99), 3)
    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
