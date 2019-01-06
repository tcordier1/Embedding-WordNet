'''
###################################################################################################

# Project: Learning Word Representations by Embedding the WordNet Graph

# Topic: Machine Learning, Natural Language Processing, Word Embeddings, Graph Embeddings

# Autors: Thibault Cordier & Antoine Tadros

###################################################################################################
'''

'''
###################################################################################################
# Step 1: Review the relevant literature on word and graph embedding methods.
###################################################################################################
'''

'''
###################################################################################################
# Step 2: Construct similarity graphs over the 80k WordNet noun synsets, using various synset similarity algorithms.
###################################################################################################
'''

'''
## Importing NLTK and WordNet
'''

import nltk

download_brown_ic = True
download_semcor_ic = False
download_genesis_ic = False

# WordNet: Import the NLTK corpus reader.
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

'''
## Importing NetworkX and librairies
'''

import networkx as nx

from tqdm import tqdm
from time import time

import numpy as np
from numpy.random import choice

import os

'''
## Extraction of WordNet noun synsets
'''

# Definition of all WordNet noun synsets
wn_all = list(wn.all_synsets('n'))
N_all = len(wn_all)

# Sample of all WordNet noun synsets
N_sel = N_all
wn_sel = wn_all

try :
    os.mkdir('graph')
    os.mkdir('emb')
except:
    pass

'''
## Construction of similarity graphs
'''

node_file = open("graph/wordnet.nodes", "w")
wn_dict = {}
for i, node in enumerate(wn_sel) :
    node_file.write(str(i)+" "+node.name()+"\n")
    wn_dict[node.name()] = i
node_file.close()

# Definition of similarity graph
sim_measures = ["naif"]

for method in sim_measures :

    print("##############################")
    print("Use method "+ method + ":")

    nx_G = nx.Graph()

    print("Add Nodes ...")

    for i, synset in enumerate(wn_sel):
        nx_G.add_node(i, synset=synset)

    print("Add Edges ...")

    for i1, synset1 in tqdm(enumerate(wn_sel), desc="Iteration over WordNet", total=N_sel):
        neighbourhood = synset1.hyponyms() + synset1.hypernyms()
        for synset2 in neighbourhood :
            nx_G.add_edge(wn_dict[synset1.name()],
                          wn_dict[synset2.name()])

    # Save similarity graph
    print("Save Similarity Graph ...")
    t_begin = time()
    nx.write_edgelist(nx_G,'graph/wordnet_' + method + '.graph', data=False)
    print("Total Time :", time()-t_begin)
