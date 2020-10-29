# -*- coding: utf-8 -*-
import os, sys , requests, pandas as pd, numpy as np, time, io, codecs, json, string, pickle, marshal, csv, matplotlib.pyplot as plt, re, unidecode
from flask import Flask, jsonify, make_response, request, render_template, Response
from gensim import corpora, models, similarities
from pprint import pprint
from collections import defaultdict
from gensim.similarities import Similarity
from gensim.models import Word2Vec
from gensim.test.utils import datapath, get_tmpfile
from flask_ngrok import run_with_ngrok
from gensim.corpora.textcorpus import TextCorpus
from gensim.test.utils import datapath, get_tmpfile, common_corpus
from langdetect import detect
from googletrans import Translator
import unidecode
from operator import itemgetter
import heapq, scipy.sparse, math, random
from sklearn.preprocessing import normalize, StandardScaler, scale, MinMaxScaler, MaxAbsScaler


# retourne la liste des variances de chaque dimension d'une matrice dense ou sparse
def get_variances(dense_or_sparse):

    variances = []
    for r in range(dense_or_sparse.shape[1]):
        col = dense_or_sparse[:,r]
        N = col.shape[0]
        donnee = col.copy()
        donnees = np.array([ pow(x,2) for x in donnee ])
        var = donnees.sum()/N - col.mean()**2
        variances.append(var)

    return variances

# retourne la liste des moyennes de chaque dimension d'une matrice dense ou sparse
def get_moyennes(dense_or_sparse):

    moyennes = []
    for r in range(dense_or_sparse.shape[1]):
        col = dense_or_sparse[:,r]
        moyennes.append(col.mean())
    moy = np.array(moyennes)

    return moy

# centrer_réduire   =>  ( "chaque_valeur" - moyenne_de_la_dimension ) / variance_de_la_dimension
                        # accepte dense_matrix ou sparse_matrix
def centrer_reduire(dense_or_sparse):

    mean = get_moyennes(dense_or_sparse)
    var = get_variances(dense_or_sparse)
    for e, sample in enumerate(dense_or_sparse):
        for f, dim in enumerate(sample):
            if dense_or_sparse[e,f] == 0.0:
                continue
            else:
                tmp = dense_or_sparse[e,f]
                dense_or_sparse[e,f] = (tmp - mean[f]) / var[f]

    return dense_or_sparse


def getVectorFromSparse(debut: int):

    '''SPARSE MATRIX (3 fichiers)
    indrp   => n° ligne
    data    => valeur
    indices => n° colonne
    '''
    sparse_data = np.load("../../../data/SparseMatrixSimilarity/full_model2/sparseFULL.index.index.data.npy")
    sparse_indices = np.load("../../../data/SparseMatrixSimilarity/full_model2/sparseFULL.index.index.indices.npy")
    sparse_indptr = np.load("../../../data/SparseMatrixSimilarity/full_model2/sparseFULL.index.index.indptr.npy")
    
    # chargement liste des numéros de messages
    # numbers = marshal.load(open("../../data/SparseMatrixSimilarity/full_model2/list_numbers", "rb"))  # désérialise et charge la liste des numero de messages
    
    interval = sparse_indptr[debut+1] - sparse_indptr[debut]
    fin = debut + interval
    valeurs = np.array([x for x in sparse_data[debut:fin]])
    vocab = np.array([x for x in sparse_indices[debut:fin]])
    vecteur = dict(zip(vocab,valeurs))

    return vecteur


# retourne le poids des mots ayant concouru � la similarit�s du message query
# index sortant = index entrant
def poids_des_mots(list_tfidf: list):
    
    tmp = [ sum(list_tfidf) - x for x in list_tfidf ]
    resultat = [ str(round((y/sum(tmp))*100, 0))+"%" for y in tmp ]

    return resultat