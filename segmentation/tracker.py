# Hand Tracker built in Python using OpenCV

import cv2
import numpy as np
import glob
import argparse
import os
import math

# http://vipulsharma20.blogspot.com/2015/03/gesture-recognition-using-opencv-python.html

# Read an image

# Grayscale and double the image

# Gaussian blur the image

# Otsu's Binarization method for thresholding to create mask

# Draw contours

# Find convex hull. The convex points are the tips of the fingers

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--image', help='image path')
	args = parser.parse_args()

	img = cv2.imread(args.image)
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	kernelSize = (35,35)
	blurred = cv2.GaussianBlur(gray_img, kernelSize, 0)

	ret2, threshold = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	# th3 = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

	# Find contours
	contours, heirarchy = cv2.findContours(threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	cnt = max(contours, key = lambda x: cv2.contourArea(x))

	x,y,w,h = cv2.boundingRect(cnt)
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),0)

	# Find the convex hull
	hull = cv2.convexHull(cnt)
	drawing = np.zeros(img.shape,np.uint8)
	cv2.drawContours(drawing,[cnt],0,(0,255,0),0)
	cv2.drawContours(drawing,[hull],0,(0,0,255),0)


	hull = cv2.convexHull(cnt,returnPoints = False)
	defects = cv2.convexityDefects(cnt,hull)

	count_defects = 0
	cv2.drawContours(threshold, contours, -1, (0,255,0), 3)
	for i in range(defects.shape[0]):
		s,e,f,d = defects[i,0]
		start = tuple(cnt[s][0])
		end = tuple(cnt[e][0])
		far = tuple(cnt[f][0])
		a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
		b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
		c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
		angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
		if angle <= 90:
			count_defects += 1
			cv2.circle(img,far,1,[0,0,255],-1)
		cv2.line(img,start,end,[0,255,0],2)
		pass

	all_img = np.hstack((drawing, img))

	cv2.imshow('image', blurred)
	cv2.imshow('threshold', threshold)
	cv2.imshow('contours', all_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()