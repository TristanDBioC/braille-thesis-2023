import cv2 as cv


def reduceResolution(image_file, resize_factor):
	output_filepath = '{}{}'.format(resize_factor, image_file)
	img = cv.imread(image_file)
	img.convertTo(image_bmp, CV_8UC3)
	res = cv.resize(img, None, fx = resize_factor, fy = resize_factor, interpolation = cv.INTER_AREA)
	cv.imwrite(output_filepath, res)

	return


image = 'alistair.png'

reduceResolution(image, 0.5)
reduceResolution(image, 0.25)
reduceResolution(image, 0.125)
reduceResolution(image, 0.0625)
