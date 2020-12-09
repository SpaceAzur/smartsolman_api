import requests, json, os, sys, marshal, pickle, time
from collections import defaultdict

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~ SAISIR NUMERO DE MESSAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
num_message = 69546
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# RECUPERATION DU N° DE MESSAGE
try:    # le n° de message peut être passé en argument du programme
    if sys.argv[1]:
        num_message = sys.argv[1] 
except: # ou saisi dans le programme dans la variable "num_message"
    num_message = num_message

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FONCTION : connection à un service ODATA pour récupérer un dictionnaire JSON
# PARAM : TYPE      NOM             CONTENU
#         string    url             adresse_HTML
#         string    identifiant     identifiant (ex: i000000234)
#         string    pwd             mot_de_passe
# RETURN : dict (dictionnaire de donnees)
def getDictFromOdata(url, identifiant, pwd):
    donnees = requests.get(url, auth=(identifiant,pwd))
    global statut 

    statut = donnees.status_code
    if (donnees.status_code != 200):
        print("erreur de connection", donnees.status_code)
        sys.exit(1)
    dict_json = donnees.json() ; del donnees
    return dict_json

def getDictFromOdata2(url, identifiant, pwd):
    donnees = requests.get(url, auth=(identifiant,pwd))
    global statut 
    statut = donnees.status_code
    if (donnees.status_code != 200):
        print("erreur de connection", donnees.status_code)
        sys.exit(1)
    dict_json = donnees.json() ; del donnees
    dico = {}
    for mes in dict_json['d']['results']:
        dico['message'] = mes['Number']
        dico['texte'] = mes['ShortText'] + ' ' + mes['LongText']
    return dico

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# REQUETAGE / EXTRACTION
# constante de connexion
statut = 200
identifiant = 'i000000275'  # VOTRE ID SOLMAN
pwd = 'RapidPareto001!'     # VOTRE MOT DE PASSE SOLMAN

# composition de l'url à requêter
u = "http://support.pasapas.com/sap/opu/odata/sap/zbow_srv/Messages?$filter=Number%20eq%20%27"
r = str(num_message)
l = "%27&$format=json"
requete = u+r+l

# initialisation d'un dictionnaire vierge
dico = defaultdict(dict)

# requêtage ODATA
data = getDictFromOdata2(requete, identifiant, pwd)

dico = dict(data)

# nommage du fichier
fichier = r+".json"

# # sauvegarde du fichier dans dossier courant
# with open(fichier,'w') as f:
#     json.dump(dico, f)


print(data)