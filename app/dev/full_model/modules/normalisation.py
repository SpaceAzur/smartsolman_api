# -*- coding: utf-8 -*-
import os, sys , requests, pandas as pd, numpy as np, time, io, codecs, json, string, pickle, marshal, csv, matplotlib.pyplot as plt, re, unidecode, Levenshtein
from gensim import corpora, models, similarities
from collections import defaultdict
from operator import itemgetter
import heapq, scipy.sparse, math, random
from sklearn.preprocessing import normalize, StandardScaler, scale, MinMaxScaler, MaxAbsScaler
from langdetect import detect
from googletrans import Translator

al = "abcdefghijklmnopqrstuvwxyz"
ALPHABET = [l for l in al]

# ratio de Levenshtein
RATIO = 0.9

# base de lemmatisation francaise
lemma = json.load(open("../../../../data/frLemmaBase_unicode_val_in_keys.json"))
lemma_values = set(lemma.values())

# base de lemmatisation anglaise
lemma_eng = json.load(open("../../../../data/EngLemmaBase_unicode.json"))

# stopwords fr
with open("../../../../data/stopwords_v1.2","rb") as f:
    stopwords_fr = pickle.load(f)

# stopwords eng
with open("../../../../data/english_stopwords_merge_of_nltk_spacy_gensim","r") as f:
    eng_stopwords = json.load(f)

# termes spécifiques SAP
with open("../../../../data/SAPtables","rb") as f:
    SAPtables = pickle.load(f)

# dictionnaire de mot unique
with open("../../../../data/dev/SparseMatrixSimilarity/full_model2/full_dico_unique","r") as u:
    dico_unique = json.load(u).keys()

with open("../../../../data/src/traductions/sap_sterm/monogram/fr_monogram.json","r") as f:
    fr_monogram = json.load(f)

with open("../../../../data/src/traductions/sap_sterm/monogram/eng_monogram.json","r") as f:
    eng_monogram = json.load(f)

with open("../../../../data/src/fautes_orthographe/dev/dict_fautes_orthographes_92.json","r") as f:
    correction_ortho = json.load(f)

# Création du dictionnaire de mot unique à un message
def getUniqueWords(corpus_normalized: list, numbers: list):
    compteur = 0
    inventaire = {}

    # compte le nombre de message solman où apparait chaque token
    for i, texte in enumerate(corpus_normalized):                              
        for token in set(texte):
            if token in inventaire.keys():
                inventaire[token] = {"count": inventaire[token].get("count") + 1 , "orig": numbers[i]}
            else:
                inventaire[token] = {"count": 1, "orig": numbers[i]}                                                       

    # sauvegarde le dictionnaire total du modele en JSON 
    with open("../../../../data/dev/SparseMatrixSimilarity/full_model2/full_dico",'w') as f:      # je sauvegarde le dictionnaire du corpus
        json.dump(inventaire,f)

    # identifie les token qui n'apparaissent que dans un seul message solman
    unique = {}
    for i, val in inventaire.items():                                        
        if val.get('count') == 1:
            unique[i] = val.get('orig')

    # sauvegarde les mots uniques en JSON
    with open("../../../../data/dev/SparseMatrixSimilarity/full_model2/full_dico_unique",'w') as f:     # je sauvegarde le dictionnaire de mot unique du corpus
        json.dump(unique,f)

    # sauvegarde en CSV
    with open("../../../../data/dev/SparseMatrixSimilarity/full_model2/dict_unique.csv", "w") as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in unique.items():
            writer.writerow([key, value])
    
    return unique
    
# décompose et recompose les termes contenant certains type de metacaractere
def transformMeta(token: str):
    global lemma
    global lemma_eng
    global SAPtables
    # with open("../../../../data/SAPtables","rb") as f:
    #     SAPtables = pickle.load(f)
    # lemma = json.load(open("../../../../data/frLemmaBase_unicode.json"))
    tmp = [ re.split(r'[\-\_]',x) for x in token.split() ]
    tmp = [ item for sublist in tmp for item in sublist ]
    lemmaRemoved = [ lemma[y] if y in lemma.keys() else y for y in tmp ]
    recompose_without_digit = "".join([ t for t in lemmaRemoved if t not in lemma.values() and not t.isdigit() and not t in SAPtables ])    
    saptable = " ".join([ x for x in lemmaRemoved if x in SAPtables ]) # recupere nom de table si inclus dans le mot (postule qu'il ne peut y avoir qu'un seul nom de table dans un mot composé)
    if saptable:
        return recompose_without_digit, saptable
    else:
        return recompose_without_digit

# LORS DU TRAITEMENT D'UN MESSAGE QUERY
# utilise la distance de Levenshtein sur les clés du dictionnaire de lemmatisation pour potentiellement corriger les fautes d'orthographes de chaque token
# return : type tuple (message corrigé: list, sous_dico_intermediaire: dict)
def checkCorrection(token):
    
    global lemma
    global lemma_eng
    global stopwords_fr
    global eng_stopwords
    global dico_unique
    global RATIO

    # On créé une liste de candidat (distance du candidat > RATIO) pour corriger une potentielle faute d'orthographe
    dis_levenshtein =[]
    for lem in lemma.keys():
        distance = Levenshtein.ratio(token.lower(), lem)
        if distance > RATIO:
            dis_levenshtein.append((lem, distance))
    dis_levenshtein.sort(key=lambda x: x[1], reverse = True)
    
    # si aucune distance > RATIO n'a été trouvé OU que token est un lemme OU que token est unique
    if ( token in lemma.keys() or token in dico_unique or len(dis_levenshtein) == 0 ):
        return token
    else: # Sinon on renvoi la correction ayant la plus courte distance de Levenshtein
        return dis_levenshtein[0] 

# corrige faute d'orthographe par calcul distance de levenshtein, algo itératif sans sauvegarde de liste
def correctionOrthographe(token):
    global lemma
    global lemma_eng

    best_distance = 0.0

    for lem in lemma.values():
        current = Levenshtein.ratio(token, lem)
        if current > best_distance:
            best_distance = current
            save = (lem, best_distance)
    
    if best_distance > RATIO:
        with open("../../../../data/dev/SparseMatrixSimilarity/full_model2/analyse_des_corrections.json","a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow((token, save[0], save[1]))
            f.close()
        return save[0]
    else:
        return token

# SANS SEUIL | recherche la meilleure distance entre toute
# corrige faute d'orthographe par calcul distance de levenshtein, algo itératif sans sauvegarde de liste
def distanceLevenshtein(token):
    global lemma
    global fautes_deja_corrigees
    best_distance = 0.0

    # with open("../../../../data/dev/SparseMatrixSimilarity/full_model2/analyse_des_corrections.json","r") as f:
    #     fautes_deja_corrigees = json.load(f)

    # si token est déjà corrigé, retourne le token
    # if token in fautes_deja_corrigees.keys():
    #     return fautes_deja_corrigees[token].get("corr")

    # pour chaque mot correctement orthographié du dictionnaire de lemmatisation en français (les clés ont été ajouté aux valeurs)
    for lem in lemma.values():
        # je calcule la distance de levenshtein
        currentz = Levenshtein.ratio(token, lem)
        current = round(currentz,5)
        # je sauvegarde la meilleure distance de manière itérative
        if current >= best_distance:
            best_distance = current
            save = (lem, best_distance)
        else:
            continue
    # fautes_deja_corrigees[token] = { "corr": save[0], "lev": save[1] }

    # with open("../../../../data/dev/SparseMatrixSimilarity/full_model2/analyse_des_corrections.json","w") as f:
    #     json.dump(fautes_deja_corrigees,f)
    
    return save
    
def apostropheX92(message: str):
    res = { re.sub(r'[\x92\n\t\r]', ' ', message) }
    res = str(res)
    res = res[2:-2]
    return res

def pasapasLemmatisation(message: list):

    global fr_monogram
    global eng_monogram 
    global correction_ortho 
    global lemma
    global lemma_eng
    global stopwords_fr
    global eng_stopwords
    global SAPtables
    global dico_unique 

    meta = re.compile(r'[-_]')
    start_with_z = re.compile(r'^z.+')
    digit_in_token = re.compile(r'\d')

    message9 = []
    for m in message:

        if m in lemma.keys():
            message9.append(lemma[m])

        elif m in lemma_eng.keys():
            message9.append(lemma_eng[m])

        elif ( meta.search(m) 
            or start_with_z.search(m)
            or digit_in_token.search(m) ):
            message9.append(m)

        elif ( m in fr_monogram.keys() or m in eng_monogram.keys() ):
            message9.append(m)
        
        elif m in correction_ortho.keys():
            correct = correction_ortho[m]['corr']
            if correct in lemma.keys():
                message9.append(lemma[correct])
            elif correct in lemma_eng.keys():
                message9.append(lemma_eng[correct])
            else:
                message9.append(correct)
            message9.append(correct)

    return message9

def pap_normalize(message: str):

    global lemma
    global lemma_eng
    global stopwords_fr
    global SAPtables
    global eng_stopwords
    global dico_unique
    global ALPHABET

    lemma_fr_values = set(lemma.values())
    lemma_eng_values = set(lemma_eng.values())

    # regex: suppression du format apostrophe r'\x92' <=> u'\u0092' (apparait comme un mini rectangle)
    x92 = re.compile(r'[\x92\']')
    if x92.search(message):
        message0 = apostropheX92(message)
    else:
        message0 = message

    # supprime \t\n\r supprime ponctuation + split les mots contenant des metacaractere qui bascule alors dans le "traitement standard"
    ponctu = re.compile(r"[\s\.,:\';°\*%><\<\>=\+\?&~#\$£€\{\}\(\)\[\]\|/\\]") 
    message1 = ponctu.sub(" ", message0)
    
    # regex : definit les mots commençant par un "z"
    start_with_z = re.compile(r'^z.+')

    # regex : définit les mots contenant au moins un chiffre
    digit_in_token = re.compile(r'\d')

    # supprime tous type d'accent (unicode)
    message2 = unidecode.unidecode(message1)

    # regex pour numero de telephone (format français)
    phone_in_token = re.compile(r'\(?(?:^0[1-9]\d{8}$|0[1-9]([\s\.\-\_]\d{2}){4}$)\)?')
    phone_in_string = re.compile(r'(?:(\d{2})?-?(\d{2})[-_\s\.]?)?(\d{2})[-_\s\.]?-?(\d{2})[-_\s\.]?-?(\d{2})[-_\s\.]?(\d{2})\)?')

    # supprime num_telephone du texte avant tokenisation
    message2bis = re.sub(phone_in_string, '', message2)

    # lowercase et tokenise (convertit en liste) OK
    message3 = re.sub(r"[\n\r\t\a]",' ', message2bis).lower() # doublon avec ligne 230 => '\s' s'en occupe déjà
    message4 = message3.split()

    # nettoyage
    while '' in message4:
        message4.remove('')

    # enlève les mots composés exclusivement de caractère non alpha-numérique (ex: "--->", "==>", "*%__") OK
    non_alfanum = re.compile(r'^\W+$')

    # supprime les mots contenant meta ["!", "§", "@", "`"]
    bob = re.compile(r'[§!@`]')
    message5 = [ i for i in message4 if not bob.search(i) and not non_alfanum.search(i) ]

    # regex : définit le pattern apostrophe
    apostr = re.compile(r"'")
    message5bis = []
    for m in message5:
        if apostr.search(m):
            un = re.sub(apostr, ' ', m).split()
            deux = [ e for e in un if not e in ALPHABET if not e in stopwords_fr if not e in eng_stopwords]
            if len(deux) == 0:
                pass
            else:
                # print(m, un, deux, len(deux))
                message5bis.append(deux[0])
        else:
            message5bis.append(m)

    meta = re.compile(r'[-_]')
    message6 = [ transformMeta(x) if meta.search(x) else x for x in message5bis ]

    # suite à 'transformMeta', des caratères peuvent être renvoyés comme termes (term) ou tuple_de_terme (term1,term2)
    # ici au besoin, nous re-splitons les tuples_de_terme en terme
    message7 = []
    for k, token in enumerate(message6):
        if not isinstance(token, tuple):
            message7.append(token)
        else:
            for y in token:
                message7.append(y)

    # suppression des horaires (ex: 4h, 12:50, 15H05)
    horaires = re.compile(r'^\d{1,2}[:hH]\d{0,2}$')
    message7bis = [ z for z in message7 if not horaires.search(z) ]

    # enlève les stopwords
    message8 = [ x for x in message7bis if not x in stopwords_fr if not x in eng_stopwords ]

    # lemmatisation, sauf si token contient un metacaractere recomposé ou commence par 'z'
    message9 = pasapasLemmatisation(message8)

    # pour être sûr, seconde suppression des stopwords 
    message10 = [ t for t in message9 if not t in stopwords_fr if not t in eng_stopwords ]

    # nettoyage des tokens vides
    while '' in message10:
        message10.remove('')

    # ????????????????? pourquoi encore
    message11 = []
    for m in message10:
        message11.append(re.sub(r'\W', '', str(m)))

    return message11

# avec traiement OK des apostrophes
def pap_normalize2(message: str):

    global lemma
    global lemma_eng
    global stopwords_fr
    global SAPtables
    global eng_stopwords
    global dico_unique
    global ALPHABET

    lemma_fr_values = set(lemma.values())
    lemma_eng_values = set(lemma_eng.values())

    # regex: suppression du format apostrophe r'\x92' <=> u'\u0092' (apparait comme un mini rectangle)
    x92 = re.compile(r'[\x92\']')
    if x92.search(message):
        message0 = apostropheX92(message)
    else:
        message0 = message

    # supprime \t\n\r supprime ponctuation + split les mots contenant des metacaractere qui bascule alors dans le "traitement standard"
    ponctu = re.compile(r"[\s\.,:\';°\*%><\<\>=\+\?&~#\$£€\{\}\(\)\[\]\|/\\]") 
    message1 = ponctu.sub(" ", message0)
    
    # regex : definit les mots commençant par un "z"
    start_with_z = re.compile(r'^z.+')

    # regex : définit les mots contenant au moins un chiffre
    digit_in_token = re.compile(r'\d')

    # supprime tous type d'accent (unicode)
    message2 = unidecode.unidecode(message1)

    # regex pour numero de telephone (format français)
    phone_in_token = re.compile(r'\(?(?:^0[1-9]\d{8}$|0[1-9]([\s\.\-\_]\d{2}){4}$)\)?')
    phone_in_string = re.compile(r'(?:(\d{2})?-?(\d{2})[-_\s\.]?)?(\d{2})[-_\s\.]?-?(\d{2})[-_\s\.]?-?(\d{2})[-_\s\.]?(\d{2})\)?')

    # supprime num_telephone du texte avant tokenisation
    message2bis = re.sub(phone_in_string, '', message2)

    # lowercase et tokenise (convertit en liste) OK
    message3 = re.sub(r"[\n\r\t\a<>]", ' ', message2).lower() # doublon avec ligne 230 => '\s' s'en occupe déjà

    
    # if x92.search(message3) != None:
    #     message3bis = apostropheX92(message3)
    #     print("in function", message3bis)
    # else:
    message3bis = message3

    # message3bis = { 'to': re.sub(r'\x92', ' ', message3) }

    # print(type(message3bis))
    # print(type(message3bis['to']))
    # message3tris = message3bis['to']

    message4 = message3bis.split(" ")

    # nettoyage
    while '' in message4:
        message4.remove('')

    # enlève les mots composés exclusivement de caractère non alpha-numérique (ex: "--->", "==>", "*%__") OK
    non_alfanum = re.compile(r'^\W+$')

    # supprime les mots contenant meta ["!", "§", "@", "`"]
    bob = re.compile(r'[§!@`]')
    message5 = [ i for i in message4 if not bob.search(i) and not non_alfanum.search(i) ]

    # 2eme traitement des apostrophes ayant pu échapper au 1er traitement 
    apostr = re.compile(r"'")
    message5bis = []
    for m in message5:
        if apostr.search(m):
            un = re.sub(apostr, ' ', m).split()
            deux = [ e for e in un if not e in ALPHABET if not e in stopwords_fr if not e in eng_stopwords]
            if len(deux) == 0:
                pass
            else:
                # print(m, un, deux, len(deux))
                message5bis.append(deux[0])
        else:
            message5bis.append(m)

    meta = re.compile(r'[-_]')
    message6 = [ transformMeta(x) if meta.search(x) else x for x in message5bis ]

    # suite à 'transformMeta', des caratères peuvent être renvoyés comme termes (term) ou tuple_de_terme (term1,term2)
    # ici au besoin, nous re-splitons les tuples_de_terme en terme
    message7 = []
    for k, token in enumerate(message6):
        if not isinstance(token, tuple):
            message7.append(token)
        else:
            for y in token:
                message7.append(y)

    # suppression des horaires (ex: 4h, 12:50, 15H05)
    horaires = re.compile(r'^\d{1,2}[:hH]\d{0,2}$')
    message7bis = [ z for z in message7 if not horaires.search(z) ]

    # enlève les stopwords
    message8 = [ x for x in message7bis if not x in stopwords_fr if not x in eng_stopwords ]

    # lemmatisation, sauf si token contient un metacaractere recomposé ou commence par 'z'
    message9 = []
    for m in message8:
        if ( meta.search(m) or start_with_z.search(m) ):
            message9.append(m)
        elif m in lemma.keys():
            message9.append(lemma[m])
        elif m in lemma_eng.keys():
            message9.append(lemma_eng[m])
        else:
            message9.append(m)


    # # correction orthographe (par distance de Levenshtein), sauf si le token est un lemme, sauf si le token contient un metacaractere recomposé, sauf si token est un terme SAP
    # message9bis = []
    # for mot in message9:
    #     if (mot in set(lemma.values()) 
    #         # or mot in lemma.keys() 
    #         # or mot in lemma_eng.keys()
    #         or mot in set(lemma_eng.values())
    #         or meta.search(mot) 
    #         or mot in SAPtables 
    #         or mot.isdigit() 
    #         # or mot in stopwords_fr 
    #         # or mot in eng_stopwords 
    #         or digit_in_token.search(mot)
    #         or start_with_z.search(mot) ):

    #         message9bis.append(mot)

    #     elif mot in fautes_deja_corrigees.keys():
    #         message9bis.append(fautes_deja_corrigees[mot].get('corr'))

    #     else:
    #         correction, distance = distanceLevenshtein(mot)
    #         message9bis.append(correction)
    #         fautes_deja_corrigees[mot] = { "corr": correction, "lev": distance }
    #         with open("../../../../data/dev/SparseMatrixSimilarity/very_light/analyse_des_corrections.json","w") as f:
    #             json.dump(fautes_deja_corrigees,f)
    #         del correction, distance

    # pour être sûr, seconde suppression des stopwords 
    message10 = [ t for t in message9 if not t in stopwords_fr if not t in eng_stopwords ]

    # nettoyage des tokens vides
    while '' in message10:
        message10.remove('')

    # ????????????????? pourquoi encore
    message11 = []
    for m in message10:
        message11.append(re.sub(r'\W', '', str(m)))

    # suppression des mots à trop forte frequence | s'execute selon si le modele est déjà créé ou pas (considération pour les appels du client)
    # çàd : cette partie s'exécute pour la normalisation du message query
    # try:
    #     with open("../../../../data/dev/SparseMatrixSimilarity/full_model2/frequence","r") as f:
    #         frequence = json.load(f)

    #     messsage12 = []
    #     for u in message11:
    #         try:
    #             z = frequence[u]
    #             if z < 20:
    #                 message12.append(u)
    #             else:
    #                 continue
    #         except:
    #             continue
    # except:
    #     pass

    # # regex : sauvegarde du pattern "n°OT"
    # # OT = re.compile(r'\w{3}k\d{6}')

    # if 'message13' in locals():
    #     return message12
    # else:
    return message11