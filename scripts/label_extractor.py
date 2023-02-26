import json



def getLabels(imagepath):
	with open('..\\new_labels.json') as f:
		data = json.load(f)

		return data[imagepath]


print(getLabels('C:\\Users\\USER\\Documents\\0 SCHOOL\\braille-thesis-2023\\preprocessed\\training\\C\\10ABC.jpg'))