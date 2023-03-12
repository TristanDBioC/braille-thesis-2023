import json
import os

def getAllContractionKeys():
	with open('..\\contractions.json', 'r') as f:
		data = json.load(f)
	keys = list(data.keys())
	return keys;


def getExpandedWord(key):
	with open('..\\contractions.json', 'r') as f:
		data = json.load(f)
		return data[key]


def replaceSubstring(main_string, old_substring, new_substring):
	return main_string.replace(old_substring, new_substring)

def parseContractions(input_string):
	keys = getAllContractionKeys()

	output_string = input_string
	for key in keys:
		output_string = replaceSubstring(output_string, key, getExpandedWord(key))
	return output_string


test_string = "bahagi ng ktr ang ccs"

print(parseContractions(test_string))

# output:
# bahagi ng kultura natin si santo niño
