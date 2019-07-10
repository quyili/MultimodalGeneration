# -*- coding: utf-8 -*-
import cv2

imgfile = "full_x.jpg"

img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)

gray = cv2.GaussianBlur(img, (3, 3), 0)

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (255, 255, 255), thickness=-1)

cv2.imwrite("mask.tiff", img)
