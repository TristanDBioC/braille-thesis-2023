import os

def filename_generator(original_path, suffix, data_type, iteration):
	output_dir = '..\\preprocessed\\{}\\{}\\'.format(data_type, iteration)
	try:
		os.makedirs(output_dir)
	except:
		pass
	extension = original_path[-4:]
	filename = os.path.basename(original_path[:-4])
	output_file = output_dir + filename + suffix + extension

	return output_file


def 