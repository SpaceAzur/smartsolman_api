# -*- coding: utf-8 -*-
import requests, json, sys
from collections import defaultdict


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FONCTION : connection à un service ODATA pour récupérer un dictionnaire JSON
# PARAM : TYPE      NOM             CONTENU
#         string    url             adresse_HTML
#         string    identifiant     identifiant (ex: i000000234)
#         string    pwd             mot_de_passe
# RETURN : dict (dictionnaire de donnees)
#@timing
def getDictFromOdata(url, identifiant, pwd):
    donnees = requests.get(url, auth=(identifiant,pwd))
    if (donnees.status_code != 200):
        print("erreur de connection", donnees.status_code)
        sys.exit(1)
    dict_json = donnees.json() ; del donnees
    return dict_json


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FONCTION : alimente le dictionnaire en token depuis le dictionnaire ODATA
# PARAM : dict()
# RETURN : dict() | dictionnaire de token
#@timing
def dictionary_to_token(dictionary):
    dico_output = defaultdict(dict)
    for message in dictionary['d']['results']:
        dico_output[message['Number']]['header'] = normalize(message.get('ShortText'))
        dico_output[message['Number']]['corps'] = normalize(message.get('LongText'))
    dico = dict(dico_output)
    return dico


def getDictFromOdata2(url, identifiant, pwd):
    donnees = requests.get(url, auth=(identifiant,pwd))
    global statut 
    statut = donnees.status_code
    if (donnees.status_code != 200):
        print("erreur de connection", donnees.status_code)
        sys.exit(1)
    dict_json = donnees.json() ; del donnees
    # print(dict_json['d']['results'])
    dico = {}
    for mes in dict_json['d']['results']:
        dico['message'] = mes['Number']
        dico['texte'] = mes['ShortText'] + ' ' + mes['LongText']
    return dico

def getURLoneMessage(num_message):
    u = "http://support.pasapas.com/sap/opu/odata/sap/zbow_srv/Messages?$filter=Number%20eq%20%27"
    r = str(num_message)
    l = "%27&$format=json"
    requete = u+r+l
    return requete

def getDictFromMessage(dict_json):
    dico = {}
    for mes in dict_json['d']['results']:
        dico['message'] = mes['Number']
        dico['texte'] = mes['ShortText'] + ' ' + mes['LongText']
    return dico