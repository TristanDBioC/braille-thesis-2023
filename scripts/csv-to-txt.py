import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches



imagePath = '..\\preprocessed\\training\\C\\'
imageFile = '1ABC.jpg'


train = pd.read_csv('..\\labels.csv')
train.head()

data = pd.DataFrame()
data['format'] = train['image_path']


for i in range(data.shape[0]):
    data['format'][i] = 'train_images/' + data['format'][i]

for i in range(data.shape[0]):
    data['format'][i] = data['format'][i] + ',' + str(train['xmin'][i]) + ',' + str(train['ymin'][i]) + ',' + str(train['xmax'][i]) + ',' + str(train['ymax'][i]) + ',' + train['braille_name'][i]

data.to_csv('..\\keras-frcnn-master\\labels.txt', header=None, index=None, sep=' ')