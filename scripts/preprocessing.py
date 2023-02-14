import cv2 as cv
import numpy as np
import os

# used to systematically generate a filename
def filepathGenerator(original_path, data_type, iteration):
	output_dir = '..\\preprocessed\\{}\\_All\\'.format(data_type, iteration)
	try:
		os.makedirs(output_dir)
	except:
		pass
	extension = original_path[-4:]
	filename = os.path.basename(original_path[:-4])
	output_file = output_dir + filename + iteration + extension
	print("Saving file:",output_file)
	return output_file


# Reduce the resolution of an image by a factor
def reduceResolution(image_file, resize_factor, data_type, iteration):
	output_filepath = filepathGenerator(image_file, data_type, iteration)

	img = cv.imread(image_file)
	res = cv.resize(img, None, fx = resize_factor, fy = resize_factor, interpolation = cv.INTER_AREA)
	cv.imwrite(output_filepath, res)
	return output_filepath



# Converting an image to greyscale
def toGrayscale(image_file, data_type, iteration):
	output_filepath = filepathGenerator(image_file, data_type, iteration)

	img = cv.imread(image_file)
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	cv.imwrite(output_filepath, gray)
	return output_filepath



def gaussianSmoothing(image_file, data_type, iteration):
	output_filepath = filepathGenerator(image_file, data_type, iteration)
	img = cv.imread(image_file)
	blur = cv.GaussianBlur(img,(3,3),0)
	cv.imwrite(output_filepath,blur)
	return output_filepath

def imageBinaryThresholding(image_file, threshold_value, max_val,  data_type, iteration):
	output_filepath = filepathGenerator(image_file, data_type, iteration)
	img = cv.imread(image_file, 0)
	ret, thresh = cv.threshold(img, threshold_value, max_val, cv.THRESH_BINARY)
	cv.imwrite(output_filepath,thresh)
	return output_filepath

def imageAdaptiveGaussThresh(image_file, max_val, size, compensation,  data_type, iteration):
	output_filepath = filepathGenerator(image_file, data_type, iteration)
	img = cv.imread(image_file, 0)
	thresh = cv.adaptiveThreshold(img, max_val, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, size, compensation)
	cv.imwrite(output_filepath,thresh)
	return output_filepath




#defining the directory of the raw data
directory = "../junk"


for filename in os.listdir(directory):
	f = os.path.join(directory, filename)
	f = reduceResolution(f, 0.25, 'training', 'A')
	f = toGrayscale(f, 'training', 'B')
	f = gaussianSmoothing(f, 'training', 'C')
	for i in [3, 5, 7, 9, 11, 13]:
		for j in range(1, 15):
			imageAdaptiveGaussThresh(f, 255, i, j, 'training', 'D_{}_{}'.format(i,j))