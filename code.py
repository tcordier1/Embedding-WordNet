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

# Information Content: Load an information content file from the wordnet_ic corpus.
# Download Brown IC
if download_brown_ic or download_semcor_ic :
    nltk.download('wordnet_ic')
    from nltk.corpus import wordnet_ic
    ic  = wordnet_ic.ic('ic-brown.dat')

# Information Content: Load an information content file from the wordnet_ic corpus.
# Download Semcor IC
if download_semcor_ic :
    nltk.download('wordnet_ic')
    from nltk.corpus import wordnet_ic
    ic = wordnet_ic.ic('ic-semcor.dat')

# Or you can create an information content dictionary from a corpus (or anything that has a words() method)
# Create Genesis IC
if download_genesis_ic :
    nltk.download('genesis')
    from nltk.corpus import genesis
    ic = wn.ic(genesis, False, 0.0)

'''
## Importing NetworkX
## and Construct similarity graphs
'''

import networkx as nx
import numpy as np
from tqdm import tqdm
from time import time

noun_list = list(wn.all_synsets('n'))
N_noun = len(noun_list)

t0 = time()
t1 = time()

nx_G = nx.Graph()

print("Add Nodes:")
for i, synset in enumerate(noun_list):
    nx_G.add_node(i)

print("Add Edges:")
'''
for i1, synset1 in enumerate(noun_list):
    if i1%round(5*N_noun/100) == 0 :
      percent = 100.*i1/N_noun
      print(percent,"%","Time :",time()-t1)
      t1 = time()
    for i2, synset2 in enumerate(noun_list):
        if i2>=i1 :
            #x_G.add_edge(i1, i2, weight=synset1.path_similarity(synset2))
'''
cur_noun_list = list()
for i1, synset1 in enumerate(noun_list):
    if i1%round(5*N_noun/100) == 0 :
        percent = 100.*i1/N_noun
        print(percent,"%","Time :",time()-t1)
        t1 = time()
    for i2, synset2 in enumerate(cur_noun_list):
        nx_G.add_edge(i1, i2, weight=synset1.res_similarity(synset2, ic))
    cur_noun_list.append(synset1)

# synset1.path_similarity(synset2)    # Hirst and St-Onge Similarity
# synset1.lch_similarity(synset2)     # Leacock-Chodorow Similarity
# synset1.wup_similarity(synset2)     # Wu-Palmer Similarity
# synset1.res_similarity(synset2, ic) # Resnik Similarity (brown_ic or genesis_ic)
# synset1.jcn_similarity(synset2, ic) # Jiang-Conrath Similarity (brown_ic or genesis_ic)
# synset1.lin_similarity(synset2, ic) # Lin Similarity (semcor_ic)

print(time()-t0)

nx.write_weighted_edgelist(nx_G,'graph/wordnet.graph')

'''
###################################################################################################
Step 2 and 3 bis: Direct implementation
###################################################################################################
'''
