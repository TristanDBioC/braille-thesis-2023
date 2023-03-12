import json
import csv



file = '..\\label.json'
output_file = '..\\labels.csv'

new_data = [['image_path','class','xmin','xmax','ymin','ymax']]
total_chars = 0

with open(file) as master_file:
	data = json.load(master_file)
	filepaths = []

	annotations = data['Annotations']
	for item in annotations:
		labels = item['Labels']
		rois = item['ROIs']['Contours']
		total_chars += len(labels)
		for i in range(len(labels)):
			data = [item['Path'][60:],str(labels[i]['Name']),rois[i]['Points'][0],rois[i]['Points'][2],rois[i]['Points'][1],rois[i]['Points'][3]]
			new_data.append(data)

with open(output_file, 'w', newline='') as f:
	writer = csv.writer(f)
	for row in new_data:
		writer.writerow(row)