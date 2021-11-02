import os
import string
import re
import pandas as pd
novels_dir="/home/jay/590-JL2616/HW5.0/novels"
labels = []
texts = []


def read_in_chunks(file_object, chunk_size=1024):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data

for fname in os.listdir(novels_dir):
	if fname[-4:] == '.txt':
		with open(os.path.join(novels_dir, fname)) as f:
		# 	for text in f.read().split("\n\n"):
		# 		if len(text.split())>=10:
		# 			texts.append(text)
		# 			labels.append(fname[:-4])
		# print(len(texts))

			for chunk in read_in_chunks(f):
				texts.append(chunk)
				labels.append(fname[:-4])

# function to remove special characters
def remove_special_characters(text):
    # define the pattern to keep
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]' 
    return re.sub(pat, '', text)

# function to remove numbers
def remove_numbers(text):
    # define the pattern to keep
    pattern = r'[^a-zA-z.,!?/:;\"\'\s]' 
    return re.sub(pattern, '', text)

# function to remove punctuation
def remove_punctuation(text):
    text = ''.join([c for c in text if c not in string.punctuation])
    return text


# function to remove special characters
def remove_extra_whitespace_tabs(text):
    #pattern = r'^\s+$|\s+$'
    pattern = r'^\s*|\s\s*'
    return re.sub(pattern, ' ', text).strip()

# function to remove special characters
def to_lowercase(text):
    return text.lower()

clean_texts=[]
for text in texts:
	text=remove_special_characters(text)
	text=remove_numbers(text)
	text=remove_punctuation(text)
	text=remove_extra_whitespace_tabs(text)
	text=to_lowercase(text)
	clean_texts.append(text)

# print(texts[1:100])
print(labels[1:100])

labels_map={"Fiander's Widow":0, "Monday or Tuesday":1,"The Castle of Otranto":2}
# "The Romance of Lust":3,"The World of Chance":4
for i in range(len(labels)):
	labels[i]=labels_map[labels[i]]

d={"text":clean_texts,"label":labels}
df=pd.DataFrame(d)

df.to_csv('cleaned_texts.csv',index=False)