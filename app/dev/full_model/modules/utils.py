# -*- coding: utf-8 -*-
import marshal, json, pickle, sys


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FONCTION : lit une liste d'élément depuis un fichier binaire
# PARAM : nom du fichier
# RETURN : generator
def yieldFromBinary(fichier):
    base = marshal.load(open(fichier,"rb"))
    for w in base:
        yield w
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FONCTION : lit une liste d'élément binaire depuis un fichier JSON
# PARAM : nom du fichier
# RETURN : generator
def yieldFromJSON(fichier):
    with open(fichier,'r', encoding='utf8') as f:
        tmp_base = json.load(f)
        for w in tmp_base:
            yield w