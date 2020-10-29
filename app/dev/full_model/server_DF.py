#!flask/bin/python
# -*- coding: UTF-8 -*-
from modules import sparseSimilarity, solman
import os, sys , requests, pandas as pd, numpy as np, time, io, codecs, json, string, pickle, marshal, csv, matplotlib as pyplot, re
from flask import Flask, jsonify, make_response, request, render_template, Response, session, flash
from gensim import corpora, models, similarities
from collections import defaultdict
from unidecode import unidecode
from flask_wtf.csrf import CSRFProtect, generate_csrf
from guppy import hpy
from memory_profiler import profile

VERSION = "dev_v1.0"

# https://pythonspot.com/login-authentication-with-flask/ BEST SO FAR
# https://stackoverflow.com/questions/39260241/flask-wtf-csrf-token-missing

'''
implémentation X-CSRF-TOKEN :
    tester nouvelle architecture du serveur avec lib flask_restful 
    => aide https://stackoverflow.com/questions/39260241/flask-wtf-csrf-token-missing
'''

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    return "Version full_dev sur Docker!"

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

# @profile
@app.route('/tfidf_classic/', methods=['GET','POST'])
def tfidf_classic():

    global VERSION
    version = VERSION + "_initial_v1"

    mydata = request.get_json(force=True)

    if 'd' in mydata.keys():
        data = {}
        for mes in mydata['d']['results']:
            data['message'] = mes['Number']
            data['texte'] = mes['ShortText'] + ' ' + mes['LongText']

            if mes["AdditionalTerms"]: 
                for k in mes["AdditionalTerms"]:
                    data['texte'] += ' ' + k

        doc, dfm2, tps2, df_both = sparseSimilarity.getSparseMatrixSimilarity(data)
    else:
        doc, dfm2, tps2, df_both = sparseSimilarity.getSparseMatrixSimilarity(mydata)

    df_both = df_both.to_json(orient='records')

    h = hpy()

    # print(h.heap())

    return jsonify(TEMPS=tps2, MESSAGE2=json.loads(df_both), TOKENS=doc, VERSION=version)

@app.route('/tfidf_custom/', methods=['GET','POST'])
def tfidf_custom():

    global VERSION
    version = VERSION + "_custom_v1"


    mydata = request.get_json(force=True)
   
    if 'd' in mydata.keys():
        data = {}
        for mes in mydata['d']['results']:
            data['message'] = mes['Number']
            data['texte'] = mes['ShortText'] + ' ' + mes['LongText']
        doc, dfm2, tps2, df_both = sparseSimilarity.getSparseMatrixSimilarity2(data)
    else:
        doc, dfm2, tps2, df_both = sparseSimilarity.getSparseMatrixSimilarity2(mydata)

    df_both = df_both.to_json(orient='records')

    return jsonify(TEMPS=tps2, MESSAGE2=json.loads(df_both), TOKENS=doc, VERSION=version)

# Execution du main
if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=False)
