import pandas as pd
import json


sheet_url = 'https://docs.google.com/spreadsheets/d/1u6nFcshVqvm7xuihT2iJ436e-NZOWnY7rfJw1Jzwelg/export?format=xlsx&gid=0'

contraction_df = pd.read_excel(sheet_url)

contraction_dict = {}


for i, row, in contraction_df.iterrows():
	contraction_dict[row['contraction']] = row['expanded word']


json_data = json.dumps(contraction_dict, indent=4)

with open('..\\contractions.json','w') as f:
	f.write(json_data)