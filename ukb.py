import networkx as nx
import json
import spacy
from nltk.wsd import lesk
nlp = spacy.load("en_core_web_sm")
class UKB:
    def __init__(self, ukb_graph):
        self.ukb_graph = ukb_graph

    def traditional_pagerank(self, subgraph):
        return nx.pagerank(subgraph)

    def personalized_pagerank(self, context_words):
        graph = self.ukb_graph
        for word in context_words.keys():
            print(word)
            graph.add_node(word, type='word')
            for concept in context_words[word]:
                graph.add_edge(word, concept)
        personalization = {node: 1.0 for node in graph.nodes() if graph.nodes[node].get('type') == 'word'} if len(context_words) != 0 else None
        pr = nx.pagerank(graph, max_iter = 50, personalization=personalization)
        for word in context_words:
            graph.remove_node(word)
        return pr

    def personalized_pagerank_w2w(self, target_word, context_words):
        graph = self.ukb_graph
        for word, concepts in context_words.items():
            graph.add_node(word, type='word')
            for concept in concepts:
                graph.add_edge(word, concept)
        personalization = {node: 10.0 for node in graph.nodes() if graph.nodes[node].get('type') == 'word' and node != target_word} if len(context_words) != 0 else None
        print(personalization)
        pr = nx.pagerank(graph, max_iter = 50, personalization=personalization)
        for s in context_words[target_word]:
            print(s, pr.get(s, "queva"))
        for word, concepts in context_words.items():
            if word != target_word:
                graph.remove_node(word)
        return pr

    def disambiguate_context(self, context_words, method = 1, freq = None, use_lesk=False):
        disambiguated_senses = {}
        if method == 1:
            subgraph = self.ukb_graph.subgraph([concept for concepts in context_words.values() for concept in concepts])
            pagerank_scores = self.traditional_pagerank(subgraph)
        elif method == 2:
            pagerank_scores = self.personalized_pagerank(context_words)
        elif method == 3:
            pagerank_scores= {}
            for target_word in context_words:
                pagerank_scores[target_word] = self.personalized_pagerank_w2w(target_word, context_words)
        else:
            return None
        
        for word, concepts in context_words.items():
            # Choose the concept with the highest PageRank score from each method
            if concepts != []:
                if method == 1 or method == 2:
                    if freq != None:
                        pagerank_scores_new = {key: pagerank_scores.get(key, 0)+0.1*freq[word.lower()].get(f"Lemma('{key}.{word.lower()}')", 0.1) for key in concepts}
                        print(pagerank_scores_new)
                    sense = max(concepts, key=lambda x: pagerank_scores.get(x, 0))
                    if use_lesk:
                        if pagerank_scores.get(sense, 0) == 0:
                            sense = lesk(context_words, word).name()
                elif method == 3:
                    sense = max(concepts, key=lambda x: pagerank_scores.get(word, {}).get(x, 0))
                disambiguated_senses[word] = sense
            else:
                disambiguated_senses[word] = None

        return disambiguated_senses

import networkx as nx
from nltk.corpus import wordnet as wn
import json
import nltk

def build_ukb_graph():
    ukb_graph = nx.Graph()

    for synset in wn.all_synsets():
        ukb_graph.add_node(synset.name(), type='synset')

    for synset in wn.all_synsets():
        for hypernym in synset.hypernyms():
            ukb_graph.add_edge(synset.name(), hypernym.name(), relation='hypernym')
        for hyponym in synset.hyponyms():
            ukb_graph.add_edge(synset.name(), hyponym.name(), relation='hyponym')
        for holonym in synset.member_holonyms() + synset.part_holonyms() + synset.substance_holonyms():
            ukb_graph.add_edge(synset.name(), holonym.name(), relation='holonym')
        for meronym in synset.member_meronyms() + synset.part_meronyms() + synset.substance_meronyms():
            ukb_graph.add_edge(synset.name(), meronym.name(), relation='meronym')
        for related_synset in synset.also_sees() + synset.similar_tos():
            ukb_graph.add_edge(synset.name(), related_synset.name(), relation='related')

    return ukb_graph

postags = {'ADV':"r", "NOUN":"n", "VERB":"v", "ADJ":"a", "PROPN":"n"}
def extract_context_words(sentence, nlp = nlp):
    # Assuming sentence is preprocessed and tokenized
    # Extract nouns, verbs, adjectives, and adverbs from the sentence
    # For each word, find associated synsets in WordNet
    context_words = {}
    pos = [(a.text, a.pos_) for a in nlp(sentence)]
    for word, pos_tag in pos:
        if pos_tag in postags.keys():
            synsets = wn.synsets(word, pos = postags[pos_tag])
            context_words[word] = [synset.name() for synset in synsets]
    return context_words

def load_ukb_graph(file_path):
    return nx.read_gexf(file_path)

def load_context_words(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
def load_sense_frequencies(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    import nltk
    try:
        ukb_graph = load_ukb_graph("ukb_graph.gexf")
        ukb_graph = nx.Graph(ukb_graph)
    except:
        print("Creating graph...")
        ukb_graph = build_ukb_graph()
        nx.write_gexf(ukb_graph, "ukb_graph.gexf")
    example_sentence = "find the solution to this problem"
    context_words = extract_context_words(example_sentence)

    frequencies = load_sense_frequencies("./data/word_sense_frequencies_semcor.json")
    ukb = UKB(ukb_graph)
    disambiguated_senses = ukb.disambiguate_context(context_words, method=3, freq=None, use_lesk=False)

    print(wn.synset("solution.n.05").definition())
    for word, sense in disambiguated_senses.items():
        print(f"Word: {word}, Sense: {sense}")
        print(wn.synset(sense).definition())

    print(wn.synset("05661668-n"))