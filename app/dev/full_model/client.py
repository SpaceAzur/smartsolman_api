#!flask/bin/python
# -*- coding: utf-8 -*-
from modules import utils, normalisation, sparseSimilarity, solman
import os, sys , requests, pandas as pd, numpy as np, time, io, codecs, json, string, pickle, marshal, csv, matplotlib as pyplot, re, ast
from collections import defaultdict
from gensim.similarities import Similarity


# récupère les arguments saisis par l'utilisateur
try:           
    if sys.argv[1] and sys.argv[2]:
        num_message = sys.argv[1]
        mode = sys.argv[2]
except:      
    print("Erreur. Il manque un ou des arguments : \n argument_1 : votre_numero_de_message \n argument_2  \n\t -> saisir i (version initial) \n\t -> saisir c (version custom) \n exemple : python3 client.py 67439 i")
    sys.exit(1)

# try:
#     if sys.argv[3]:
#         keyword = sys.argv[3]
# except:
#     pass

try:
    if sys.argv[3]:
        keyw = sys.argv
        keywords = keyw[3:]
except:
    pass


# numbers = marshal.load(open("../../../data/dev/SparseMatrixSimilarity/full_model2/list_numbers", "rb"))
# num_query = int(num_message)

# if num_query in numbers:
#     print("YES", numbers.index(num_query))

# constante de connexion
statut = 200
user = "i000000275"         # VOTRE ID SOLMAN
pwd = "RapidPareto001!"     # VOTRE MOT DE PASSE SOLMAN

# récupère le message query depuis solman
requete = solman.getURLoneMessage(num_message=num_message)

# agence le message en dictionnaire, traitable par le modèle
data = solman.getDictFromOdata2(url=requete, identifiant=user, pwd=pwd)


# if 'keyword' in locals():
#     data['texte'] += ' ' + keyword
# else:
#     pass

if 'keywords' in locals():
    for k in keywords:
        data['texte'] += ' ' + k
else:
    pass

# URL à requêter sur serveur.py  (sauvegarde nom d'hôte sapdocpy.pasapas.com)
if mode == 'i':
    url2 = "http://localhost:5001/tfidf_classic/"
    for_save = 'initial'
elif mode == 'c':
    url2 = "http://localhost:5001/tfidf_custom/"
    for_save = 'custom'
else:
    print("Argument {} incorrect !\n\tChoisir 'i' pour initial ou 'c' pour custom".format(mode))
    sys.exit(1)

# interroge le modèle
headers = {'Content-type': 'application/json', 'Accept': 'text/plain', 'Referer': url2}
req_serveur = requests.post(url2, data=json.dumps(data), headers=headers)

# recupère réponse en JSON
rep_json = req_serveur.json()

# formate les 3 dictionnaires contenus dans le JSON (1. les messages similaires, 2. les mots pertinents, 3. les emps d'exécution)
temps = rep_json['TEMPS']
message_bis = rep_json['MESSAGE2']
tokens = rep_json['TOKENS']

# message_bis = json.loads(message_bis)

pd.set_option('display.max_colwidth', 70) # défini une largeur de colonne pour le dataframe
dfmsg_bis = pd.DataFrame.from_dict(message_bis, orient='columns') # convertit un dictionnaire en json
suggestions = dfmsg_bis['autres suggestions'] # recupere à part la list des autres suggestions
dfmsg_bis = dfmsg_bis.drop(['SparseSimilaritY','%.','autres suggestions','Mots(count)'], axis=1) # supprime 2 colonnes en doublons

# sauvegarde pour analyse du resultat
dfmsg_bis.to_csv("test_unitaire/"+str(sys.argv[1])+for_save)

# affichage resultat
print('\nTOKENS MESSAGE {}\n'.format(str(num_message)),tokens)
print('\n',dfmsg_bis)

# mise en forme et affichage séparé de 'autres suggestions'
sugg_message = [ x[0] for x in list(suggestions) if not x is None if x[0] != int(num_message) ] # if x[0] != int(num_message)
sugg_mot = [ x[1] for x in list(suggestions) if not x is None if x[0] != int(num_message) ]
sugg = { "messages": sugg_message, "mot": sugg_mot}
suggestion = pd.DataFrame(sugg)
print('\nAUTRES SUGGESTIONS\n',suggestion)

# temps d'exécution API
print("\ntime (sec) ", round(temps,2))

# divise une liste en sous-ensemble de 10 elements
def pretty_tokens(lst):
    res = []
    for i in range(0, len(lst), 10):
        res.append(lst[i:i + 10])
    return res

# mise en forme de la sauvegarde csv
with open("test_unitaire/"+str(sys.argv[1])+for_save,'a') as f:
    f.write("\nAUTRES SUGGESTIONS\n")
    f.write(suggestion.to_string())
    f.write('\n\nTOKENS {}\n'.format(str(num_message)))
    for e in pretty_tokens(tokens):
        f.write(json.dumps(e))
        f.write('\n')

