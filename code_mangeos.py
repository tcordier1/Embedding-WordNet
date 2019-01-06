'''
###################################################################################################

# Project: Learning Word Representations by Embedding the WordNet Graph

# Topic: Machine Learning, Natural Language Processing, Word Embeddings, Graph Embeddings

# Autors: Thibault Cordier & Antoine Tadros

###################################################################################################
'''

import nltk

# WordNet: Import the NLTK corpus reader.
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

'''
## Importing NetworkX
## and Construct similarity graphs
'''

import networkx as nx

from tqdm import tqdm
from time import time

import numpy as np
import os

# Definition of all WordNet noun synsets
wn_full = list(wn.all_synsets('n'))
wn_all = wn_full
N_all = len(wn_all)

#%% Load synset embedding
with open("emb/emb_naif.emd") as file :
    _, dim = map(int, file.readline().split(' '))
    synset_emb = np.zeros((N_all, dim))
    for line in file.readlines() :
        elt = line.split(' ')
        i = int(elt[0])
        for j, nbr in enumerate(elt[1:]) :
            synset_emb[i,j] = float(nbr)

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
'''

'''
num_vertices = 82114
dim = 128
filepath = "/Users/thibaultcordier/Downloads/res_simnet_deepwalk/res_simnet_deepwalk.emb"
synset_emb = np.fromfile(filepath, np.float32).reshape(num_vertices, dim)
'''

print("matrix of shape : ", synset_emb.shape)
#print(synset_emb[82104,])

#%% get list of nouns
noun_list = np.unique([lemma.name() for synset in wn_all for lemma in synset.lemmas()])

#%% build dict of synset
syn_dict = {}
for i, syn in enumerate(wn_all):
    syn_dict[syn.name()] = i

#%% Word embedding, beware of the index for synsets (no node 0)
n_words = len(noun_list)
word_emb = np.zeros((n_words,dim))
word_dict = {}

for i, word in enumerate(noun_list):
    word_dict[word] = i
    synsets = wn.synsets(word,pos=wn.NOUN)
    synset_indices = [syn_dict[synset.name()] for synset in synsets]
    word_emb[i,:] = synset_emb[synset_indices,:].mean(axis=0)

#%% MANGEOS
import mangoes
import string

source = word_dict
matrix = word_emb

vocabulary = mangoes.Vocabulary(source=source)
embeddings = mangoes.Embeddings(vocabulary, matrix)

print(embeddings.get_closest_words("dog", 10))
print(embeddings.get_closest_words("cat", 10))
print(embeddings.get_closest_words("man", 10))
ws_result = mangoes.evaluate.similarity(embeddings)
print(ws_result.detail)
