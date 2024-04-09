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
    
    def subgraph_pagerank(self,context_words):
        disambiguation_graph = nx.Graph()
        for word, concepts in context_words.items():
            for concept in concepts:
                bfs_paths = nx.single_source_shortest_path(self.ukb_graph, concept)
                for other_word, other_concepts in context_words.items():
                    if other_word != word:
                        for other_concept in other_concepts:
                            if other_concept in bfs_paths.keys():
                                shortest_path = bfs_paths[other_concept]
                                disambiguation_graph.add_edges_from(zip(shortest_path[:-1], shortest_path[1:]))
        return nx.pagerank(disambiguation_graph)
    
    def personalized_pagerank(self, context_words):
        graph = self.ukb_graph
        for word in context_words.keys():
            graph.add_node(word, type='word')
            for concept in context_words[word]:
                graph.add_edge(word, concept)
        personalization = {n: 10.0 for node in graph.nodes() for n in graph.neighbors(node)  if graph.nodes[node].get('type') == 'word'} if len(context_words) != 0 else None
        starting = {node: 10.0 for node in graph.nodes() if graph.nodes[node].get('type') == 'word'} if len(context_words) != 0 else None
        if personalization != None and len(personalization) == 0:
            personalization = None
        pr = nx.pagerank(graph, max_iter = 30, personalization=personalization, nstart=starting)
        for word in context_words:
            graph.remove_node(word)
        return pr

    def personalized_pagerank_w2w(self, target_word, context_words, starting):
        graph = self.ukb_graph
        personalization = {n: 10.0 for node in graph.nodes() for n in graph.neighbors(node)  if graph.nodes[node].get('type') == 'word' and node != target_word} if len(context_words) > 1 else None
        if personalization != None and len(personalization) == 0:
            personalization = None
        pr = nx.pagerank(graph, max_iter = 30, personalization=personalization, nstart=starting)
        return pr

    def disambiguate_context(self, context_words, method = 1, freq = None, use_lesk=False):
        disambiguated_senses = {}
        if method == 0:
            subgraph = self.ukb_graph.subgraph([concept for concepts in context_words.values() for concept in concepts])
            pagerank_scores = self.traditional_pagerank(subgraph)
        elif method == 1:
            pagerank_scores = self.subgraph_pagerank(context_words)
        elif method == 2:
            pagerank_scores = self.personalized_pagerank(context_words)
        elif method == 3:
            pagerank_scores= {}
            for word, concepts in context_words.items():
                self.ukb_graph.add_node(word, type='word')
                for concept in concepts:
                    self.ukb_graph.add_edge(word, concept)
            starting = {node: 10.0 for node in self.ukb_graph.nodes() if self.ukb_graph.nodes[node].get('type') == 'word'} if len(context_words) != 0 else None
            for target_word in context_words:
                pagerank_scores[target_word] = self.personalized_pagerank_w2w(target_word, context_words, starting)
            for word, concepts in context_words.items():
                self.ukb_graph.remove_node(word)
        else:
            return None
        
        for word, concepts in context_words.items():
            # Choose the concept with the highest PageRank score from each method
            if concepts != []:
                if method == 0 or method == 2:
                    if freq != None:
                        pagerank_scores_new = {key: pagerank_scores.get(key, 0)+0.1*freq[word.lower()].get(f"Lemma('{key}.{word.lower()}')", 0.1) for key in concepts}
                    sense = max(concepts, key=lambda x: pagerank_scores.get(x, 0))
                    if use_lesk:
                        if pagerank_scores.get(sense, 0) == 0:
                            sense = lesk(context_words, word).name()
                elif method == 1:
                    sense = max(concepts, key=lambda x: pagerank_scores.get(x, 0))
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
    example_sentence = "mix the solution to this experiment"
    context_words = extract_context_words(example_sentence)

    frequencies = load_sense_frequencies("./data/word_sense_frequencies_semcor.json")
    ukb = UKB(ukb_graph)
    disambiguated_senses = ukb.disambiguate_context(context_words, method=1, freq=None, use_lesk=False)

    """
    print(wn.synset("solution.n.05").definition())
    for word, sense in disambiguated_senses.items():
        print(f"Word: {word}, Sense: {sense}")
        print(wn.synset(sense).definition())
    """