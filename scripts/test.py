import cv2 as cv
import numpy as np
import os
from general import filename_generator




def reduce_resolution(image_file, resize_factor, suffix, data_type, iteration):
	img = cv.imread(image_file)
	res = cv.resize(img, None, fx = resize_factor, fy = resize_factor, interpolation = cv.INTER_AREA)
	cv.imwrite(filename_generator(image_file, suffix, data_type, iteration), res)



directory = "../Raw Data/training data"


for filename in os.listdir(directory):
	f = os.path.join(directory, filename)	
	reduce_resolution(f, 0.25, 'rs', 'training',2)
	
