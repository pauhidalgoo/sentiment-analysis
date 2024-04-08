import nltk
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews as mr
import numpy as np
import json

ids_positive = mr.fileids('pos')
words_positive = [" ".join(mr.words(file)) for file in ids_positive]
ids_negative = mr.fileids('neg')
words_negative = [" ".join(mr.words(file)) for file in ids_negative]

X = words_positive + words_negative
y = np.array([1 for _ in range(len(ids_positive))] + [0 for _ in range(len(ids_negative))])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True, stratify=y)

X_train_no_val, X_val, y_train_no_val, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, shuffle=True, stratify=y_train)


# Save the datasets
with open('./data/X_train.json', 'w') as file:
    json.dump(X_train, file)

with open('./data/X_test.json', 'w') as file:
	json.dump(X_test, file)

with open('./data/y_train.json', 'w') as file:	
	json.dump(y_train.tolist(), file)

with open('./data/y_test.json', 'w') as file:
	json.dump(y_test.tolist(), file)

with open('./data/X_train_no_val.json', 'w') as file:
	json.dump(X_train_no_val, file)

with open('./data/X_val.json', 'w') as file:
	json.dump(X_val, file)

with open('./data/y_train_no_val.json', 'w') as file:
	json.dump(y_train_no_val.tolist(), file)

with open('./data/y_val.json', 'w') as file:
	json.dump(y_val.tolist(), file)
