import re, os, sys, json, subprocess, copy, pandas as pd, itertools, numpy as np, progressbar, O365
from datetime import datetime
from collections import namedtuple

# creation d'une classe de test
message_test = namedtuple("Classe_de_test",["query","attendu"])

# instanciation des constantes testées
test1 = message_test(76140,[58453, 27762, 33540])
test2 = message_test(76560,[66348])
test3 = message_test(72473,[2382, 23019, 4169])
test4 = message_test(69546,[65189])

# création de la base test
baseTest = {
			test1.query: {	"attendu": test1.attendu,
							"recu_initial": {"similarites":[], "%":[]},
							"recu_cutom": {"similarites":[], "%":[]} 
                        	},
			test2.query: {	"attendu": test2.attendu,
							"recu_initial": {"similarites":[], "%":[]},
							"recu_cutom": {"similarites":[], "%":[]} 
                        	},
			test3.query: {	"attendu": test3.attendu,
							"recu_initial": {"similarites":[], "%":[]},
							"recu_cutom": {"similarites":[], "%":[]} 
                        	},
			test4.query: {	"attendu": test4.attendu,
							"recu_initial": {"similarites":[], "%":[]},
							"recu_cutom": {"similarites":[], "%":[]} 
                        	}
				}

baseTest2 = {
			test1.query: {	"attendu": test1.attendu,
							"initial_similarites": [], 
							"initial_%":0.0,
							"custom_similarites": [], 
							"custom_%":0.0 
                        	},
			test2.query: {	"attendu": test2.attendu,
							"initial_similarites": [], 
							"initial_%":0.0,
							"custom_similarites": [], 
							"custom_%":0.0 
                        	},
			test3.query: {	"attendu": test3.attendu,
							"initial_similarites": [], 
							"initial_%":0.0,
							"custom_similarites": [], 
							"custom_%":0.0 
                        	},
			test4.query: {	"attendu": test4.attendu,
							"initial_similarites": [], 
							"initial_%":0.0,
							"custom_similarites": [], 
							"custom_%":0.0 
                        	}
				}

# copy de la base test
resultat = copy.deepcopy(baseTest)
resultat2 = copy.deepcopy(baseTest2)

# sauvegarde
with open("base_test","w") as f:
    json.dump(baseTest,f)

# chargement
with open("base_test","r") as f:
    base_test = json.load(f)

# 	"i": modèle gensim.Tfidf paramètres par défaut
# 	"c": modèle gensim.Tfidf paramètres 'smartirs'
mode = ["i","c"]

# instanciation de la bar de progression
bar = progressbar.ProgressBar(maxval=len(list(resultat.keys())*len(mode)), \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()] )
c = 0
bar.start()

# appelle du client pour chaque message test et chaque mode
for query, val in base_test.items():
	for m in mode:
		bar.update(c)
		result = os.popen("python3 masse.py {} {}".format(query, m)).read()
		res = json.loads(re.sub("'",'"', result))
		
		sugg = res['from_unique']
		sugg_msg = sugg['messages']
		sugg_mot = sugg['mot']
		suggestion = list(zip(sugg_msg, sugg_mot))

		# sauvegarde des resultats
		if m == "i":
			resultat[int(query)]["recu_initial"]["similarites"] = res["similarites"]
			resultat[int(query)]["recu_initial"]["%"] = res["%"]

			resultat2[int(query)]["initial_similarites"] = [ r for r in res["similarites"] if r in resultat2[int(query)]["attendu"] ]
			resultat2[int(query)]["initial_%"] = round(len([ r for r in res["similarites"] if r in resultat2[int(query)]["attendu"] ]) / len(resultat2[int(query)]["attendu"]),2) * 100
			resultat2[int(query)]["suggestions"] = suggestion

		elif m == "c":
			resultat[int(query)]["recu_cutom"]["similarites"] = res["similarites"]
			resultat[int(query)]["recu_cutom"]["%"] = res["%"]

			resultat2[int(query)]["custom_similarites"] = [ r for r in res["similarites"] if r in resultat2[int(query)]["attendu"] ]
			resultat2[int(query)]["custom_%"] = round(len([ r for r in res["similarites"] if r in resultat2[int(query)]["attendu"] ]) / len(resultat2[int(query)]["attendu"]),2) * 100
		c += 1
bar.finish()

# cast en dataframe
df = pd.DataFrame.from_dict(resultat, orient="index")
df2 = pd.DataFrame.from_dict(resultat2, orient="index")

current = datetime.now()
cle = current.strftime("%Y_%m_%d-%Hh%M")

df2.to_csv("{}.csv".format(cle))

print()
print(df2)
print()











# base_test = {
# 			76140: {"attendu": [58453, 27762, 33540], 
# 					"recu": {"similarites":[], "%":[]} 
# 					},
# 			76560: {"attendu": [66348], 
# 					"recu": {"similarites":[], "%":[]}
# 					},
# 			72473: {"attendu": [2382, 23019, 4169], 
# 					"recu": {"similarites":[], "%":[]}
# 					},
# 			69546: {"attendu": [65189], 
# 					"recu": {"similarites":[], "%":[]}
# 					}          
#                 }