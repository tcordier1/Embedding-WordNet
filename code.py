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

from tqdm import tqdm
from time import time

import numpy as np
from numpy.random import choice

from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix

import os

# Definition of all WordNet noun synsets
wn_all = list(wn.all_synsets('n'))
N_all = len(wn_all)
#%%
# Sample of all WordNet noun synsets
N_sel = 5000
wn_sel = choice(wn_all,N_sel,False)

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

for method in sim_measures :

    print("##############################")
    print("Use method "+ method + ":")

    # Definition of times
    t_begin = time()
    t_current = time()
    t_cumul = time()-t_begin

    nx_G = nx.Graph()
    #A = dok_matrix((N_sel,N_sel))

    print("Add Nodes ...")

    for i, synset in enumerate(wn_sel):
        nx_G.add_node(i, synset=synset)

    print("Add Edges ...")
    wn_visited = list()

#%% # k_list
 
method = 'res'
k = 100
n_nodes = len(wn_all)

#ind = np.zeros((n_nodes, k))
#weights = np.zeros((n_nodes, k))

print("##############################")
print("Use method "+ method + ":")

# Definition of times
t_begin = time()
t_current = time()
t_cumul = time()-t_begin

nx_G = nx.Graph()
#A = dok_matrix((N_sel,N_sel))

print("Add Nodes ...")

for i, synset in enumerate(wn_all):
    nx_G.add_node(i, synset=synset)

print("Add Edges ...")
wn_visited = list()

for i1, synset1 in tqdm(enumerate(wn_all)):

#    if i1%round(5*N_sel/100) == 0 and i1!=0 :
#        percent = 100.*i1/N_sel
#        t_step = time()-t_current
#        t_cumul += t_step
#        t_final = 400*t_cumul/(np.sum(np.arange(1,1+percent/5)*2-1))
#        print(percent,"%","Time Step :",t_step,"Time Cumul :",t_cumul,"Time Final :",t_final)
    nn_ind = list()
    nn_weight = list()
    nb_nn = 0
        
    for i2, synset2 in enumerate(wn_all):

        # n = 5000
        # synset1.path_similarity(synset2)    # Hirst and St-Onge Similarity >1h
        # synset1.lch_similarity(synset2)     # Leacock-Chodorow Similarity >1h10min
        # synset1.wup_similarity(synset2)     # Wu-Palmer Similarity >1h20min
        # synset1.res_similarity(synset2, ic) # Resnik Similarity (brown_ic or genesis_ic) 130 sec
        # synset1.jcn_similarity(synset2, ic) # Jiang-Conrath Similarity (brown_ic or genesis_ic) 150 sec
        # synset1.lin_similarity(synset2, ic) # Lin Similarity (semcor_ic) 130 sec
        if synset1==synset2 : 
            pass
        else :
            sim = synset1.res_similarity(synset2, ic)
            
#                
#            if method == "path" :
#                sim = synset1.path_similarity(synset2)
#            elif method == "lch" :
#                sim = synset1.lch_similarity(synset2)
#            elif method == "wup" :
#                sim = synset1.wup_similarity(synset2)
#            elif method == "res" :
#                sim = synset1.res_similarity(synset2, ic)
#            elif method == "jcn" :
#                sim = synset1.jcn_similarity(synset2, ic)
#            elif method == "lin" :
#                sim = synset1.lin_similarity(synset2, ic)
#                
            if len(nn_ind) < k :
                
                nn_ind.append(i2)
                nn_weight.append(sim)
            else : 
                if sim < np.min(nn_weight):
                    ind_min = np.argmin(nn_weight)
                    nn_weight[ind_min] = sim
                    nn_ind[ind_min] = i2
        
        
    for m in range(k):
        
        nx_G.add_edge(i1, nn_ind[m], weight=nn_weight[m])

#    wn_visited.append(synset1)

# Print total time
print("Total Time :", time()-t_begin)

# Save similarity graph
print("Save Similarity Graph ...")
t_begin = time()
#nx_G = nx.from_scipy_sparse_matrix(A)
nx.write_weighted_edgelist(nx_G,'graph/wordnet_' + method + '.graph')
print("Total Time :", time()-t_begin)
    for i1, synset1 in enumerate(wn_sel):

        if i1%round(5*N_sel/100) == 0 and i1!=0 :
            percent = 100.*i1/N_sel
            t_step = time()-t_current
            t_cumul += t_step
            t_final = 400*t_cumul/(np.sum(np.arange(1,1+percent/5)*2-1))
            print(percent,"%","Time Step :",t_step,"Time Cumul :",t_cumul,"Time Final :",t_final)
            t_current = time()

        for i2, synset2 in enumerate(wn_visited):

            # n = 5000
            # synset1.path_similarity(synset2)    # Hirst and St-Onge Similarity >1h
            # synset1.lch_similarity(synset2)     # Leacock-Chodorow Similarity >1h10min
            # synset1.wup_similarity(synset2)     # Wu-Palmer Similarity >1h20min
            # synset1.res_similarity(synset2, ic) # Resnik Similarity (brown_ic or genesis_ic) 130 sec
            # synset1.jcn_similarity(synset2, ic) # Jiang-Conrath Similarity (brown_ic or genesis_ic) 150 sec
            # synset1.lin_similarity(synset2, ic) # Lin Similarity (semcor_ic) 130 sec

            if method == "path" :
                sim = synset1.path_similarity(synset2)
            elif method == "lch" :
                sim = synset1.lch_similarity(synset2)
            elif method == "wup" :
                sim = synset1.wup_similarity(synset2)
            elif method == "res" :
                sim = synset1.res_similarity(synset2, ic)
            elif method == "jcn" :
                sim = synset1.jcn_similarity(synset2, ic)
            elif method == "lin" :
                sim = synset1.lin_similarity(synset2, ic)
            nx_G.add_edge(i1, i2, weight=sim)

        wn_visited.append(synset1)

    # Print total time
    print("Total Time :", time()-t_begin)

    # Save similarity graph
    print("Save Similarity Graph ...")
    t_begin = time()
    #nx_G = nx.from_scipy_sparse_matrix(A)
    nx.write_weighted_edgelist(nx_G,'graph/wordnet_' + method + '.graph')
    print("Total Time :", time()-t_begin)
    

#%% # k_list
 
method = 'res'
k = 100
n_nodes = len(wn_all)

#ind = np.zeros((n_nodes, k))
#weights = np.zeros((n_nodes, k))

print("##############################")
print("Use method "+ method + ":")

# Definition of times
t_begin = time()
t_current = time()
t_cumul = time()-t_begin

nx_G = nx.Graph()
#A = dok_matrix((N_sel,N_sel))

print("Add Nodes ...")

for i, synset in enumerate(wn_all):
    nx_G.add_node(i, synset=synset)

print("Add Edges ...")
wn_visited = list()

for i1, synset1 in tqdm(enumerate(wn_all)):

#    if i1%round(5*N_sel/100) == 0 and i1!=0 :
#        percent = 100.*i1/N_sel
#        t_step = time()-t_current
#        t_cumul += t_step
#        t_final = 400*t_cumul/(np.sum(np.arange(1,1+percent/5)*2-1))
#        print(percent,"%","Time Step :",t_step,"Time Cumul :",t_cumul,"Time Final :",t_final)
    nn_ind = list()
    nn_weight = list()
    nb_nn = 0
        
    for i2, synset2 in enumerate(wn_all):

        # n = 5000
        # synset1.path_similarity(synset2)    # Hirst and St-Onge Similarity >1h
        # synset1.lch_similarity(synset2)     # Leacock-Chodorow Similarity >1h10min
        # synset1.wup_similarity(synset2)     # Wu-Palmer Similarity >1h20min
        # synset1.res_similarity(synset2, ic) # Resnik Similarity (brown_ic or genesis_ic) 130 sec
        # synset1.jcn_similarity(synset2, ic) # Jiang-Conrath Similarity (brown_ic or genesis_ic) 150 sec
        # synset1.lin_similarity(synset2, ic) # Lin Similarity (semcor_ic) 130 sec
        if synset1==synset2 : 
            pass
        else :
            sim = synset1.res_similarity(synset2, ic)
            
#                
#            if method == "path" :
#                sim = synset1.path_similarity(synset2)
#            elif method == "lch" :
#                sim = synset1.lch_similarity(synset2)
#            elif method == "wup" :
#                sim = synset1.wup_similarity(synset2)
#            elif method == "res" :
#                sim = synset1.res_similarity(synset2, ic)
#            elif method == "jcn" :
#                sim = synset1.jcn_similarity(synset2, ic)
#            elif method == "lin" :
#                sim = synset1.lin_similarity(synset2, ic)
#                
            if len(nn_ind) < k :
                
                nn_ind.append(i2)
                nn_weight.append(sim)
            else : 
                if sim < np.min(nn_weight):
                    ind_min = np.argmin(nn_weight)
                    nn_weight[ind_min] = sim
                    nn_ind[ind_min] = i2
        
        
    for m in range(k):
        
        nx_G.add_edge(i1, nn_ind[m], weight=nn_weight[m])

#    wn_visited.append(synset1)

# Print total time
print("Total Time :", time()-t_begin)

# Save similarity graph
print("Save Similarity Graph ...")
t_begin = time()
#nx_G = nx.from_scipy_sparse_matrix(A)
nx.write_weighted_edgelist(nx_G,'graph/wordnet_' + method + '.graph')
print("Total Time :", time()-t_begin)

#%% # full list, the fastest
 
method = 'lch'

sim_measures = ["res","jcn","lin","path","lch","wup"]

for method in sim_measures :
        
    k = 100
    n_nodes = len(wn_all)
    
    
    print("##############################")
    print("Use method "+ method + ":")
    
    # Definition of times
    t_begin = time()
    t_current = time()
    t_cumul = time()-t_begin
    
    nx_G = nx.Graph()
    
    print("Add Nodes ...")
    
    for i, synset in enumerate(wn_all):
        nx_G.add_node(i, synset=synset)
    
    print("Add Edges ...")
    wn_visited = list()
    
    for i1, synset1 in tqdm(enumerate(wn_all)):
    
        
        
        if method == "path" :
            sim_func = synset1.path_similarity
            need_ic = False
        elif method == "lch" :
            sim_func = synset1.lch_similarity
            need_ic = False
        elif method == "wup" :
            sim_func = synset1.wup_similarity
            need_ic = False
        elif method == "res" :
            sim_func = synset1.res_similarity
            need_ic = True
        elif method == "jcn" :
            sim_func = synset1.jcn_similarity
            need_ic = True
        elif method == "lin" :
            sim_func = synset1.lin_similarity
            need_ic = True
            
        nn_ind = np.zeros(n_nodes)
        nn_weight = np.zeros(n_nodes)
        nb_nn = 0
        
        for i2, synset2 in enumerate(wn_all):
    
            # n = 5000
            # synset1.path_similarity(synset2)    # Hirst and St-Onge Similarity >1h
            # synset1.lch_similarity(synset2)     # Leacock-Chodorow Similarity >1h10min
            # synset1.wup_similarity(synset2)     # Wu-Palmer Similarity >1h20min
            # synset1.res_similarity(synset2, ic) # Resnik Similarity (brown_ic or genesis_ic) 130 sec
            # synset1.jcn_similarity(synset2, ic) # Jiang-Conrath Similarity (brown_ic or genesis_ic) 150 sec
            # synset1.lin_similarity(synset2, ic) # Lin Similarity (semcor_ic) 130 sec

            if i2==i1 : 
                pass
            else :
                if need_ic : 
                    sim = sim_func(synset2, ic)
                else : 
                    sim = sim_func(synset2)
                nn_weight[i2] = sim
                
        nn_weight[i1] = 0       # set self-similarity to zero
        ind_sort = np.argsort(nn_weight)
        knn_ind = ind_sort[-k:] # take neighbours with biggest similarity 
        nn_weight.sort()
        knn_weight = nn_weight[-k:]

        i1_list = [n for n in range(k)]
        edge_list = zip(i1_list,list(knn_ind))
        nx_G.add_edges_from(edge_list, weight=knn_weight)
    
    #    wn_visited.append(synset1)
    
    # Print total time
    print("Total Time :", time()-t_begin)
    
    # Save similarity graph
    print("Save Similarity Graph ...")
    t_begin = time()
    #nx_G = nx.from_scipy_sparse_matrix(A)
    nx.write_weighted_edgelist(nx_G,'graph/wordnet_' + method + '.graph')
    print("Total Time :", time()-t_begin)


'''
###################################################################################################
Step 2 and 3 bis: Direct implementation
###################################################################################################
'''
