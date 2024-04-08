from collections import defaultdict
import nltk
import json
import time
def extract_word_senses(tagged_sentence):
    word_senses = []
    for word_entry in tagged_sentence:
        if isinstance(word_entry, nltk.Tree):
            word_entry = word_entry.flatten()
        elif isinstance(word_entry, list):
            continue
        word = word_entry[0] if len(word_entry) == 1 else None
        if word != None:
            sense = str(word_entry.label())
            word_senses.append((word, sense))
    return word_senses

def create_sense_frequencies():
    nltk.download('semcor')
    nltk.download('wordnet')
    sense_frequencies = defaultdict(lambda: defaultdict(int))
    semcor = nltk.corpus.semcor
    for sentence in semcor.tagged_sents(tag='sem'):
        sentence = extract_word_senses(sentence)
        for word, synset in sentence:
            sense_frequencies[word][synset] += 1

    for word in sense_frequencies.keys():
        total = sum(sense_frequencies[word].values())
        for syn in sense_frequencies[word].keys():
            sense_frequencies[word][syn] = sense_frequencies[word][syn] / total

    return sense_frequencies

def save_sense_frequencies(sense_frequencies, filename):
    with open(filename, 'w') as f:
        json.dump(sense_frequencies, f)

sense_frequencies = create_sense_frequencies()
save_sense_frequencies(sense_frequencies, 'word_sense_frequencies_semcor.json')