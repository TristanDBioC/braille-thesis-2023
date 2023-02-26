import json



file = '..\\label.json'

new_dict = {'images':[{'path':'asdasd', 'labels':[{'name':'aaaa','points':[111,111,111,111]}]}]}
new_dict = {}

with open(file) as master_file:
	data = json.load(master_file)
	filepaths = []

	annotations = data['Annotations']
	for item in annotations:
		labels = []
		rois = []

		
		for label in item['Labels']:
			labels.append(label['Name'])
		for roi in item['ROIs']['Contours']:
			rois.append(roi['Points'])


		print(item['Path'][-8:])
		print(len(labels), len(rois))
		new_dict[item['Path']] = [[labels[i], rois[i]] for i in range(len(labels))]
		break




with open('..\\new_labels.json', 'w') as f:
	f.write(json.dumps(new_dict))
