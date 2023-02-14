import pandas as pd
import json


contraction_df = pd.read_excel('..\\Contraction dictionary.xlsx', sheet_name='Sheet1')

contraction_dict = {}

for i, row, in contraction_df.iterrows():
	contraction_dict[row['contraction']] = row['expanded word']


json_data = json.dumps(contraction_dict, indent=4)

with open('..\\contractions.json','w') as f:
	f.write(json_data)