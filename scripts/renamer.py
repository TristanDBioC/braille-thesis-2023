import os

directory = "../Raw Data/training data"


files = os.listdir(directory)

for i, filename in enumerate(files):
	new_filename = f'{i+1}.jpg'

	oldpath = os.path.join(directory, filename)
	newpath = os.path.join(directory, new_filename)
	os.rename(oldpath, newpath)