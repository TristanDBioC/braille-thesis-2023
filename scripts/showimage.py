import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches



imagePath = '..\\preprocessed\\training\\C\\'
imageFile = '1ABC.jpg'


train = pd.read_csv('..\\labels.csv')
train.head()

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

image = plt.imread(imagePath+imageFile)


for _,row in train[train.image_path == imageFile].iterrows():
	xmin = row.xmin
	xmax = row.xmax
	ymin = row.ymin
	ymax = row.ymax

	width = xmax - xmin
	height = ymax - ymin
	ax.annotate(str(row.braille_name),xy=(xmin,ymin))

	rect = patches.Rectangle((xmin,ymin), width, height, edgecolor='r', facecolor = 'none')
	ax.add_patch(rect)

plt.imshow(image)
plt.show()