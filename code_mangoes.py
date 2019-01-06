'''
###################################################################################################

# Project: Learning Word Representations by Embedding the WordNet Graph

# Topic: Machine Learning, Natural Language Processing, Word Embeddings, Graph Embeddings

# Autors: Thibault Cordier & Antoine Tadros

###################################################################################################
'''

import os
import argparse

import nltk
import networkx as nx
import numpy as np

from tqdm import tqdm
from time import time

import mangoes
import string

# WordNet: Import the NLTK corpus reader.
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

def parse_args():
    '''
    Parses the code mangoes arguments.
    '''
    parser = argparse.ArgumentParser(description="Run mangoes test.")

    parser.add_argument('--input', nargs='?', default='emb/emb.emd',
        help='Input embedding path')
    parser.add_argument('--debug', dest='debug', action='store_true',
        help='Launch Debug. Default is False.')
    parser.add_argument('--test', dest='test', action='store_false')
    parser.set_defaults(debug=False)

    return parser.parse_args()

def load_wordnet() :
    # Definition of all WordNet noun synsets
    print("Load WordNet ...")
    wn_all = list(wn.all_synsets('n'))
    N_all = len(wn_all)

    # Build list of nouns
    idx2noun = np.unique([lemma.name() for synset in wn_all for lemma in synset.lemmas()])

    # Build dict of synset
    syn2idx = {}
    for i, syn in enumerate(wn_all):
        syn2idx[syn.name()] = i

    return idx2noun, syn2idx

def load_embedding_synset(path, dim_wn) :
    # Load synset embedding
    with open(path) as file :
        _, dim_emb = map(int, file.readline().split(' '))
        synset_emb = np.zeros((dim_wn, dim_emb))
        for line in tqdm(file.readlines(), desc="Load Synset Embedding") :
            elt = line.split(' ')
            i = int(elt[0])
            for j, nbr in enumerate(elt[1:]) :
                synset_emb[i,j] = float(nbr)

    return synset_emb

def load_embedding_noun(synset_emb, idx2noun, syn2idx) :
    #%% Word embedding, beware of the index for synsets (no node 0)
    n_synsets, dim = synset_emb.shape
    n_words = len(idx2noun)
    word_emb = np.zeros((n_words,dim))

    word2idx = {}
    for i, word in tqdm(enumerate(idx2noun), desc="Load Word Embedding", total=len(idx2noun)):
        word2idx[word] = i
        synsets = wn.synsets(word,pos=wn.NOUN)
        synset_indices = [syn2idx[synset.name()] for synset in synsets]
        word_emb[i,:] = synset_emb[synset_indices,:].mean(axis=0)

    return word2idx, word_emb

'''
num_vertices = 82114
dim = 128
filepath = "/Users/thibaultcordier/Downloads/res_simnet_deepwalk/res_simnet_deepwalk.emb"
synset_emb = np.fromfile(filepath, np.float32).reshape(num_vertices, dim)
'''

def test(emb) :

    '''
    L1 = []
    for i, synset1 in enumerate(wn_all) :
         if synset1.hyponyms() + synset1.hypernyms() == [] :
              L1.append(i)
    L2 = [i for i in range(N_all)]
    with open("emb/emb_naif.emd") as file :
        _, dim = map(int, file.readline().split(' '))
        synset_emb = np.zeros((N_all, dim))
        for line in file.readlines() :
            elt = line.split(' ')
            i = int(elt[0])
            L2.remove(i)
            for j, nbr in enumerate(elt[1:]) :
                synset_emb[i,j] = float(nbr)

    S1 = set(L1)
    print(L1[:10])

    L2 = sorted(L2)
    S2 = set(L2)
    print(L2[:10])

    print(len(list(S2-S1)))

    print(synset_emb[82104,])
    '''

    print(emb.get_closest_words("dog", 10))
    print(emb.get_closest_words("cat", 10))
    print(emb.get_closest_words("man", 10))

def main(args):

    idx2noun, syn2idx = load_wordnet()
    synset_emb = load_embedding_synset(args.input, len(syn2idx))
    source, matrix = load_embedding_noun(synset_emb, idx2noun, syn2idx)

    vocabulary = mangoes.Vocabulary(source)
    embeddings = mangoes.Embeddings(vocabulary, matrix)

    if args.debug :
        test(embeddings)

    ws_result = mangoes.evaluate.similarity(embeddings)
    print(ws_result.detail)

if __name__ == "__main__":
	args = parse_args()
	main(args)
