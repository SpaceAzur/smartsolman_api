import os, sys , requests, pandas as pd, numpy as np, time, io, codecs, json, string, pickle, marshal, csv, matplotlib.pyplot as plt, re, unidecode
from gensim import corpora, models, similarities
from collections import defaultdict
from langdetect import detect
from googletrans import Translator
from operator import itemgetter
import heapq, scipy.sparse, math, random
from guppy import hpy
# from memory_profiler import memory_usage, profile

sys.path.append('.')
from modules.normalisationdatabase import NormalisationDataBase
from modules.normalisation2 import Normalisation2
# from modules import normalisation


def getSparseMatrixSimilarity(dico_element: dict): # INITIAL

    db = NormalisationDataBase()
    normaliser = Normalisation2(db)

    t1 = time.time()
    
    dictionary = corpora.Dictionary.load('/smartsolman/data/dev/SparseMatrixSimilarity/full_model2/specifiques.dict') #   chargement du dictionnaire

    # SPARSE MATRIX
    data = np.load("/smartsolman/data/dev/SparseMatrixSimilarity/full_model2/classic_model/sparse.index.index.data.npy")
    colonnes = np.load("/smartsolman/data/dev/SparseMatrixSimilarity/full_model2/classic_model/sparse.index.index.indices.npy")
    lignes = np.load("/smartsolman/data/dev/SparseMatrixSimilarity/full_model2/classic_model/sparse.index.index.indptr.npy")
    
    sparseMatrix = scipy.sparse.csr_matrix((data, colonnes, lignes), shape=(len(lignes)-1,len(dictionary)))
    
    corpus = corpora.MmCorpus('/smartsolman/data/dev/SparseMatrixSimilarity/full_model2/specifiques.mm')     # chargement du corpus 
    num = dico_element['message']
    mes = dico_element['texte']     # récupération du texte à comparer dans la base

    doc = normaliser.pap_normalize(mes)
    # doc = normaliser.papapas_normalize(mes)    # normalisation + lemmatisation + tokenisation 
    vec_bow = dictionary.doc2bow(doc)   # vectorisation en compteur de mot du message input
    
    # tfidf = models.TfidfModel(corpus=corpus, normalize=True)   # génère modele Tfidf du corpus
    
    tfidf_loaded = models.TfidfModel.load("/smartsolman/data/dev/SparseMatrixSimilarity/full_model2/classic_model/tfidf_classic.model")

    vec_new_msg = tfidf_loaded[vec_bow]    # vectorisation en TF-IDF du message input
    query_sparse_vec = vec_new_msg  # sauvegarde du vecteur du message query
    index = similarities.SparseMatrixSimilarity.load('/smartsolman/data/dev/SparseMatrixSimilarity/full_model2/classic_model/sparse.index')    # désérialise et charge la matrice d'index des similarités
    numbers = marshal.load(open("/smartsolman/data/dev/SparseMatrixSimilarity/full_model2/list_numbers", "rb"))  # désérialise et charge la liste des numero de messages
    resultat = index[vec_new_msg] # recherche de similarités
    sims = sorted(enumerate(resultat), key=lambda item: -item[1])   # classement des similarités
    vec_new_msg = sorted(vec_new_msg, key=itemgetter(1), reverse=True)
    vec_bow = sorted(vec_bow, key=itemgetter(1), reverse=True)
    with open("/smartsolman/data/dev/SparseMatrixSimilarity/full_model2/full_dico_unique","r") as f:     # chargement dict mot unique
        unique = json.load(f)

    memoire = {'matrice': sys.getsizeof(sparseMatrix),
                'data': sys.getsizeof(data),
                'lignes': sys.getsizeof(lignes),
                'colonnes': sys.getsizeof(colonnes),
                'vec_bow': sys.getsizeof(vec_bow),
                'vec_nex_msg': sys.getsizeof(vec_new_msg),
                'dictionnaire': sys.getsizeof(dictionary),
                'modele': sys.getsizeof(tfidf_loaded),
                'index': sys.getsizeof(tfidf_loaded),
                'corpus': sys.getsizeof(corpus),
                'numbers': sys.getsizeof(numbers),
                'resultat': sys.getsizeof(resultat),
                'sims': sys.getsizeof(sims),
                'dict_unique': sys.getsizeof(unique)
                }
    # dmem = pd.DataFrame.from_dict(memoire)
    # dmem.to_excel("memoireusage.xlsx")
    # with open("memoireusage.json","w") as f:
    #     json.dump(memoire,f)

    # h = hpy()

    # print(h.heap())

    # Récupère les 10 messages les plus similaires
    q = 0
    resul = [] # LISTE DES NUMEROS DE MESSAGES LES PLUS SIMILAIRES
    while q < 10:
        val = sims[q][0]
        # print(val, sims[q])
        resul.append(numbers[val])
        q += 1
  
    # Tri décroissant les coef de similarité, pour affichage json
    resultat = sorted(resultat, reverse=True)

    msg_similaires = []
    g = 0
    while g != len(resul):
        msg_similaires.append((resul[g], round((resultat[g] * 100), 2)))
        g += 1
    
    # A JARTER----------------------------------------------
    # liste des mots les plus pertinent
    mots = [ dictionary[w[0]] for w in vec_new_msg ]

    # construction du resultat (list of tuple ( 10 messages + similaires, % de similarité pour chacun, 3 mots les plus pertinents pour chacun ) )
    synthese = []
    cpt = 0
    for sim in [ x[0] for x in sims[0:11] ]:                # pour chacun des 10 messages les + similaires
        foo = []
        for word in [ y[0] for y in corpus[sim] ]:          # pour chaque mot du message
            if word in [ z[0] for z in vec_bow[0:10] ] :    # recherche des mots qui ont été pertinent 
                foo.append(dictionary[word])        
        synthese.append((numbers[sim], round(sims[cpt][1]*100, 1), foo[0:3]))  # 
        cpt += 1  

    dfm2 = pd.DataFrame(synthese, index=None, columns=['SparseSimilarity','%','Mots(count)'])

    # conversion du message_query : tuple to array  
    query_array = np.zeros(len(dictionary))
    for i, val in query_sparse_vec:
        query_array[i] = val

    # je calcul la norme du vecteur query
    norme = np.linalg.norm(query_array)

    # je normalise le vecteur du message_query
    n_query_array = query_array / norme

    synthese = []
    cpt = 0
    mots_perti3 = [] 
    for sim in [ x[0] for x in sims[0:11] ]: # pour chacun des 10 messages les plus similaires
        mots_perti = []     # je créé un liste1
        mots_perti2 = []    # je créé une liste2
        
        # je calcule la norme du vecteur du message courant
        current_norme = np.linalg.norm(sparseMatrix[sim].toarray())

        # je normalise le vecteur du messaage courant
        current_vector = sparseMatrix[sim].toarray() / current_norme

        # je fais le produit du msg similaire avec le msg_query
        compar = current_vector * n_query_array 

        # je lisse le vecteur (list of list => list)
        compar = compar.flatten() 

        # je récupère l'indice des 3 plus grandes valeurs
        res = compar.argsort(axis=None)[-3:][::-1]
        
        # je caste le resultat en liste
        resu = list(res)

        # je construit la liste des 3 mots les plus pertinent
        mots_pertinents = [ (round(compar[res[m]],4), dictionary[res[m]]) for m, val in enumerate(resu) ]

        # ANCIENNNE VERSION --------------------------------------------------------------------------------------------------------------------
                # for i, val in enumerate(query_sparse_vec): # pour chaque élément du vecteur composant le message_query
                    
                #     # if sparseMatrix[sim,val[0]] == 0.0: # si la valeur de l'élément est zéro
                #     #     pass                            # alors je l'ignore
                #     # else:                               # sinon
                #     #     mots_perti.append(( val[0], abs(sparseMatrix[sim,val[0]] / np.linalg.norm(sparseMatrix[sim].toarray()) - val[1] / np.linalg.norm([ i[1] for i in query_sparse_vec ]))))
                #     #     mots_perti2.append(( val[0], abs(sparseMatrix[sim,val[0]] - val[1])) )
                    
                #     # mots_perti.append(( val[0],    ))
                #     mots_perti.append(( val[0], abs(sparseMatrix[sim,val[0]] / np.linalg.norm(sparseMatrix[sim].toarray()) - val[1] / np.linalg.norm([ i[1] for i in query_sparse_vec ]))))
                #     mots_perti2.append(( val[0], abs(sparseMatrix[sim,val[0]] - val[1])) )

                # mots_perti.sort(key=lambda item: item[1]) # tri croissant
                # mots_perti2.sort(key=lambda item: item[1]) # tri croissant
                
                # synthese.append((numbers[sim], round(sims[cpt][1]*100,1), [(dictionary[t[0]], round(t[1],4)) for t in mots_perti[0:3]], [(dictionary[t[0]], round(t[1],4)) for t in mots_perti2[0:3]] ))
                # synthese.append((numbers[sim], round(sims[cpt][1]*100,1), [(dictionary[t[0]], round(t[1],4)) for t in mots_perti[0:3]]))
        # ---------------------------------------------------------------------------------------------------------------------------------------


        # j'alimente la liste de resultats des 10 messages similaires qui sera affichée
        synthese.append((numbers[sim], round(sims[cpt][1]*100,1), mots_pertinents ))
        cpt += 1
    
    # df_tfidf_rescaled = pd.DataFrame(synthese, index=None, columns=['SparseSimilaritY','%.','mots_TFIDF', 'mot_TFIDF non_normé'])
    df_tfidf_rescaled = pd.DataFrame(synthese, index=None, columns=['SparseSimilaritY','%.','mots_TFIDF'])

    df_both = pd.concat([dfm2,df_tfidf_rescaled], axis=1)

    # Champ SUGGESTION
    suggestion = set()
    for i, token in enumerate(doc):
        if token in unique.keys():
            suggestion.add((unique[token],token))

    suggestion = list(suggestion)
    df_both.loc[:,'autres suggestions'] = pd.Series(suggestion[0:10])

    # suppression du message query de l'affichage des similarités
    top10 = df_both["SparseSimilarity"].tolist()
    if int(num) in top10:
        ig = top10.index(int(num))
        df_both = df_both.drop(ig, axis=0)

    # dataframe
    columns = ['index dict','occur']
    dfw = pd.DataFrame(vec_bow, index=mots, columns=columns).nlargest(10, columns='occur')
    dfm = pd.DataFrame(msg_similaires, index=None, columns=['Matrix_Msg','Matrix_%'])
    t2 = time.time()
    temps = t2 - t1
    db.close()
    return doc, dfm2, temps, df_both

def getSparseMatrixSimilarity2(dico_element: dict): # CUSTOM

    db = NormalisationDataBase()
    normaliser = Normalisation2(db)

    t1 = time.time()
    #TF-IDF : MatrixSimilarity
    
    dictionary = corpora.Dictionary.load('/smartsolman/data/dev/SparseMatrixSimilarity/full_model2/specifiques.dict') #   chargement du dictionnaire

    with open("/smartsolman/data/dev/SparseMatrixSimilarity/full_model2/frequence","r") as f:
        frequence = json.load(f)

    # SPARSE MATRIX
    data = np.load("/smartsolman/data/dev/SparseMatrixSimilarity/full_model2/custom_model/lfn/sparse.index.index.data.npy")
    colonnes = np.load("/smartsolman/data/dev/SparseMatrixSimilarity/full_model2/custom_model/lfn/sparse.index.index.indices.npy")
    lignes = np.load("/smartsolman/data/dev/SparseMatrixSimilarity/full_model2/custom_model/lfn/sparse.index.index.indptr.npy")
    
    sparseMatrix = scipy.sparse.csr_matrix((data, colonnes, lignes), shape=(len(lignes)-1,len(dictionary)))
    
    corpus = corpora.MmCorpus('/smartsolman/data/dev/SparseMatrixSimilarity/full_model2/specifiques.mm')     # chargement du corpus 
    num = dico_element['message']
    mes = dico_element['texte']     # récupération du texte à comparer dans la base
    doc = normaliser.pap_normalize(mes)
    vec_bow = dictionary.doc2bow(doc)   # vectorisation en compteur de mot du message input

    tfidf_loaded = models.TfidfModel.load("/smartsolman/data/dev/SparseMatrixSimilarity/full_model2/custom_model/lfn/tfidf_custom.model")

    vec_new_msg = tfidf_loaded[vec_bow]    # vectorisation en TF-IDF du message input
    query_sparse_vec = vec_new_msg  # sauvegarde du vecteur du message query
    index = similarities.SparseMatrixSimilarity.load('/smartsolman/data/dev/SparseMatrixSimilarity/full_model2/custom_model/lfn/sparse.index')    # désérialise et charge la matrice d'index des similarités
    numbers = marshal.load(open("/smartsolman/data/dev/SparseMatrixSimilarity/full_model2/list_numbers", "rb"))  # désérialise et charge la liste des numero de messages
    resultat = index[vec_new_msg] # recherche de similarités
    sims = sorted(enumerate(resultat), key=lambda item: -item[1])   # classement des similarités
    vec_new_msg = sorted(vec_new_msg, key=itemgetter(1), reverse=True)
    vec_bow = sorted(vec_bow, key=itemgetter(1), reverse=True)
    with open("/smartsolman/data/dev/SparseMatrixSimilarity/full_model2/full_dico_unique","r") as f:     # chargement dict mot unique
        unique = json.load(f)

    # Récupère les 10 messages les plus similaires
        # q = 0
        # resul = [] # LISTE DES NUMEROS DE MESSAGES LES PLUS SIMILAIRES
        # while q < 10:
        #     val = sims[q][0]
        #     print("val", val, sims[q])
        #     resul.append(numbers[val])
        #     q += 1
  
    # Tri décroissant les coef de similarité, pour affichage json
    resultat = sorted(resultat, reverse=True)

    # msg_similaires = []
        # g = 0
        # while g != len(resul):
        #     msg_similaires.append((resul[g], round((resultat[g] * 100), 2)))
        #     g += 1
    
    # liste des mots les plus pertinent
    mots = [ dictionary[w[0]] for w in vec_new_msg ]

    # construction du resultat (list of tuple ( 10 messages + similaires, % de similarité pour chacun, 3 mots les plus pertinents pour chacun ) )
    synthese = []
    cpt = 0
    for sim in [ x[0] for x in sims[0:11] ]:                # pour chacun des 10 messages les + similaires
        foo = []
        for word in [ y[0] for y in corpus[sim] ]:          # pour chaque mot du message
            if word in [ z[0] for z in vec_bow[0:10] ] :    # recherche des mots qui ont été pertinent 
                foo.append(dictionary[word])        
        synthese.append((numbers[sim], round(sims[cpt][1]*100, 1), foo[0:3]))  # 
        cpt += 1  

    dfm2 = pd.DataFrame(synthese, index=None, columns=['SparseSimilarity','%','Mots(count)'])

    # conversion du message_query : tuple to array  
    query_array = np.zeros(len(dictionary))
    for i, val in query_sparse_vec:
        query_array[i] = val

    # je calcul la norme du vecteur query
    norme = np.linalg.norm(query_array)

    # je normalise le vecteur du message_query
    n_query_array = query_array / norme

    synthese = []
    cpt = 0
    mots_perti3 = [] 
    for sim in [ x[0] for x in sims[0:11] ]: # pour chacun des 10 messages les plus similaires
        mots_perti = []     # je créé un liste1
        mots_perti2 = []    # je créé une liste2
        
        # je calcule la norme du vecteur du message courant
        current_norme = np.linalg.norm(sparseMatrix[sim].toarray())

        # je normalise le vecteur du messaage courant
        current_vector = sparseMatrix[sim].toarray() / current_norme

        # je fais le produit du msg similaire avec le msg_query
        compar = current_vector * n_query_array 

        # je le lisse en une seule dimension
        compar = compar.flatten() 

        # je récupère l'indice des 3 plus grandes valeurs
        res = compar.argsort(axis=None)[-3:][::-1]
        
        # je caste le resultat en liste
        resu = list(res)

        # je construit la liste des 3 mots les plus pertinent
        mots_pertinents = [ (round(compar[res[m]],4), dictionary[res[m]]) for m, val in enumerate(resu) ]

        # j'alimente la liste de resultats des 10 messages similaires qui sera affichée
        synthese.append((numbers[sim], round(sims[cpt][1]*100,1), mots_pertinents ))
        cpt += 1
    
    df_tfidf_rescaled = pd.DataFrame(synthese, index=None, columns=['SparseSimilaritY','%.','mots_TFIDF'])

    df_both = pd.concat([dfm2,df_tfidf_rescaled], axis=1)

    # Champ SUGGESTION
    suggestion = set()
    for i, token in enumerate(doc):
        if token in unique.keys():
            suggestion.add((unique[token],token))

    suggestion = list(suggestion)
    # print(pd.Series(suggestion[0:10]))
    df_both.loc[:,'autres suggestions'] = pd.Series(suggestion[0:10])

    # suppression du message query dans les résultats de similarités
    top10 = df_both["SparseSimilarity"].tolist()
    if int(num) in top10:
        ig = top10.index(int(num))
        df_both = df_both.drop(ig, axis=0)

    # dataframe
    columns = ['index dict','occur']
    dfw = pd.DataFrame(vec_bow, index=mots, columns=columns).nlargest(10, columns='occur')
    # dfm = pd.DataFrame(msg_similaires, index=None, columns=['Matrix_Msg','Matrix_%'])
    t2 = time.time()
    temps = t2 - t1
    db.close()
    return doc, dfm2, temps, df_both

def toto(x,p):
    return math.pow(x,p)

def myNorme(vecteur):
    return math.sqrt(sum(list(map(lambda v: math.pow(v,2), vecteur))))

def bothNotNull(valeur1, valeur2):
    return valeur1 != 0.0 and valeur2 != 0.0

def approach(vecteur1, vecteur2):
    return np.array(list(map(lambda x: abs(x[0]-x[1]) if bothNotNull(x[0], x[1]) else None, zip(vecteur1, vecteur2))))

