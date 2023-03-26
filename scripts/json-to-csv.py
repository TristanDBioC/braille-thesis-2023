import json
import csv



file = '..\\test.json'
output_file = '..\\test.csv'

new_data = [['image_path','label','xmin','ymin','xmax','ymax']]
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
			data = ['test_images\\'+item['Path'][60:],str(labels[i]['Name']),rois[i]['Points'][0],rois[i]['Points'][1],rois[i]['Points'][2],rois[i]['Points'][3]]
			new_data.append(data)

with open(output_file, 'w', newline='') as f:
	writer = csv.writer(f)
	for row in new_data:
		writer.writerow(row)