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
## Importing NetworkX and librairies
'''

import networkx as nx

from tqdm import tqdm
from time import time

import numpy as np
from numpy.random import choice

from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix

import os


'''
## Extraction of WordNet noun synsets
'''

# Definition of all WordNet noun synsets
wn_all = list(wn.all_synsets('n'))
N_all = len(wn_all)
#%%
# Sample of all WordNet noun synsets
N_sel = N_all #1000
wn_sel = wn_all #choice(wn_all,N_sel,False)

try :
    os.mkdir('graph')
    os.mkdir('emb')
except:
    pass

'''
## Construction of similarity graphs
'''

try :
 os.mkdir('graph')    
except:
    pass

node_file = open("graph/wordnet.nodes", "w")
for i, node in enumerate(wn_sel) :
    node_file.write(str(i)+" "+node.name()+"\n")
node_file.close()

# Definition of similarity graph
sim_measures = ["path","lch","wup","res","jcn","lin"]
sim_measures = ["res"]
k = 100

for method in sim_measures :

    print("##############################")
    print("Use method "+ method + ":")

    nx_G = nx.Graph()

    print("Add Nodes ...")

    for i, synset in enumerate(wn_sel):
        nx_G.add_node(i, synset=synset)

    print("Add Edges ...")


    for i1, synset1 in tqdm(enumerate(wn_sel), desc="Iteration over WordNet", total=N_sel):

        nn_ind = [] #np.zeros(k)
        nn_weight = [] #np.zeros(k)
        nb_nn = 0
        min_weight = +np.Inf
        min_ind = i1

        # n = 5000
        # synset1.path_similarity(synset2)    # Hirst and St-Onge Similarity >1h
        # synset1.lch_similarity(synset2)     # Leacock-Chodorow Similarity >1h10min
        # synset1.wup_similarity(synset2)     # Wu-Palmer Similarity >1h20min
        # synset1.res_similarity(synset2, ic) # Resnik Similarity (brown_ic or genesis_ic) 130 sec
        # synset1.jcn_similarity(synset2, ic) # Jiang-Conrath Similarity (brown_ic or genesis_ic) 150 sec
        # synset1.lin_similarity(synset2, ic) # Lin Similarity (semcor_ic) 130 sec

        if method == "path" :
            sim_func = lambda synset2 : synset1.path_similarity(synset2)
        elif method == "lch" :
            sim_func = lambda synset2 : synset1.lch_similarity(synset2)
        elif method == "wup" :
            sim_func = lambda synset2 : synset1.wup_similarity(synset2)
        elif method == "res" :
            sim_func = lambda synset2 : synset1.res_similarity(synset2, ic)
        elif method == "jcn" :
            sim_func = lambda synset2 : synset1.jcn_similarity(synset2, ic)
        elif method == "lin" :
            sim_func = lambda synset2 : synset1.lin_similarity(synset2, ic)

        for i2, synset2 in enumerate(wn_sel):

            try :
                sim = sim_func(synset2)
            except :
                sim = 0
            if sim > 0 :
                if nb_nn < k :
                    nn_ind.append(i2)
                    nn_weight.append(sim)
                    nb_nn += 1
                    if sim <= min_weight :
                        min_weight = sim
                        min_ind = i2
                else :
                    if sim <= min_weight :
                        ()
                    else :
                        nn_ind.append(i2)
                        nn_weight.append(sim)
                        idx = nn_ind.index(min_ind)
                        del nn_weight[idx]
                        del nn_ind[idx]
                        min_weight = np.min(nn_weight)
                        min_idx = nn_weight.index(min_weight)
                        min_ind = nn_ind[min_idx]

        knn_ind = nn_ind
        knn_weight = nn_weight

        i1_list = [i1 for n in range(k)]
        edge_list = zip(i1_list,list(knn_ind),knn_weight)
        nx_G.add_weighted_edges_from(edge_list)

    # Save similarity graph
    print("Save Similarity Graph ...")
    t_begin = time()
    nx.write_weighted_edgelist(nx_G,'graph/wordnet_' + method + '.graph')
    print("Total Time :", time()-t_begin)
