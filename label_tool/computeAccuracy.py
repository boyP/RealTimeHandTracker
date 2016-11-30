# Labeling tool for creating the hand tracking dataset for Dranimate 

# Program reads two directories of textfiles containing
# the ground truth created using the labler.py and the estimates using a hand
# tracking algorithm of your choice and then it calculates the error using 
# sum of squared differences. 

# The text files should contain 5 (x,y) coordinates of the fingers in this order 
#   (x1,y1) => pinky 
#   (x2,y2) => ring 
#   (x3,y3) => middle 
#   (x4,y4) => index 
#   (x5,y5) => thumb

#
# To run in command line: 
# python computeAccuracy.py --input1 <InputDir_1> --input2 <InputDir_2>
# Ex. python computeAccuracy.py --input1 <path/to/ground_truth/> --input2 <path/to/estimate>
#
#

import cv2
import numpy as np
import glob
import argparse
import os

def computeAccuracy(ground_truth, estimate):
	return np.linalg.norm(estimate - ground_truth)

#### MAIN ####
parser = argparse.ArgumentParser()
parser.add_argument('--input1', help='textfile input 1 directory')
parser.add_argument('--input2', help='textfile input 2 directory')
args = parser.parse_args()

totalErr = 0

for textPath in glob.glob(args.input1 + "*.txt"):
	# Get the same filename in directory 2
	fileName = os.path.basename(textPath)
	textPath2 = args.input2 + fileName;

	if not os.path.isfile(textPath2):
		raise IOError(textPath2 + ' estimate file not found')

	# Load text files to compare
	truth    = np.loadtxt(textPath)
	estimate = np.loadtxt(textPath2)
	if len(truth) != 5 or len(estimate) != 5:
		raise ValueError(fileName + ' does not contain 5 points')

	err = computeAccuracy(truth, estimate)
	totalErr += err

print('Error: ' + str(totalErr))
