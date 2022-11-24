##Ranking basado en el sentimiendo promedio de cada video en funcion de los comentarios de los usuarios.

import pandas as pd
import numpy as np

from pymongo import MongoClient
MONGO_URI = "mongodb://localhost"

client = MongoClient(MONGO_URI)

dbTest = client["teststore"] #si no tengo la db mongo la crea

collection = dbTest["u2comentarios"]


######

import nltk
import spacy
from nltk.tokenize.treebank import TreebankWordDetokenizer

from pysentimiento import create_analyzer

from cleantext import clean
import re

import matplotlib.pyplot as plt
import json
import string
from nltk.corpus import stopwords

#nltk.download('stopwords')
#spacy.cli.download("es_core_news_md")
#nlp = spacy.load("es_core_news_md")

#######LECTURA DE JSON OBTENIDO DE MONGODB, LIMPIEZA DE COMENTARIO, LEMATIZACIÓN E INSERTAR EN MONGODB
"""
def lematiza(comment):
    lista = []
    #text = i["comentario"]
    comment = ' '.join(re.sub("(@[A-Za-z0-9]+)", " ", comment).split())
    comment = ' '.join(re.sub("(\w+:\/\/\S+)", " ", comment).split())
    comment = ' '.join(re.sub("[\.\,\«\»\!\¡\[\]\{\}\'\?\¿\:\;\+\^\-\=]", " ", comment).split())
    # text = ' '.join(re.sub(":D", " ", text).split())
    # text = ' '.join(re.sub(":d", " ", text).split())
    # text = ' '.join(re.sub(":\u263a", " ", text).split())
    comment = ' '.join(re.sub("\d+", " ", comment).split())
    comment = clean(comment, no_emoji=True)  # si quita emoticones
    tokens_spacy = nlp(comment)
    for w in tokens_spacy:
        lista.append(w.lemma_)
    return lista


with open('u2comentariosPreProce.json', 'r', encoding="utf8") as jsonFile:
    jsonObject = json.load(jsonFile)
    jsonFile.close()

list_n = []
list_id = []
list_comment = []
c = 0
for i in jsonObject:
    c += 1
    #print("n:",c,"id_video:", i["id_video"], ", comentario:", i["comentario"])
    list_n.append(c)
    list_id.append(i["id_video"])
    list_comment.append(i["comentario"])

#print("#lista n")
#print(list_n)
print("cantidad de n", len(list_n))
#print("#lista id")
#print(list_id)
print("cantidad de ids", len(list_id))
#print("#lista comentarios")
#print(list_comment)
print("cantidad de comentarios", len(list_comment))

cuenta = 0
comentario_lematizado = []
for i in list_comment:
    cuenta += 1
    print("n:", cuenta, "comentario:", i)
    comentario_lematizado.append(lematiza(i))


cuenta2 = 0
#lematiza()
print("Lista lematizada")
for i in comentario_lematizado:
    cuenta2 += 1
    print("n:", cuenta2, "tokens:",i)
"""
#almancenar en mongodb en un nuevo campo la lista de tokens lematizados y limpiados de cada comentario
"""
cursor = collection.find({})
for comen,lista_tokenizada in zip(cursor,comentario_lematizado):
    collection.update_one(comen, {'$set': {'tokenizado': lista_tokenizada}})
"""
#####FIN#################
##STOP WORDS Y DESTOKENIZACION - sentimiento+insertar en mongodb####
"""
contador = 0
for x in collection.find():
    contador += 1
    print("N:",contador, "Tokenizado;",x["tokenizado"])
"""

"""
stopwords_esp_spacy = nlp.Defaults.stop_words
stopwords_completo_es = set()
stopwords_completo_es.update(stopwords_esp_spacy)
stopwords_completo_es.update(set(stopwords.words('spanish')))
stopwords_completo_es.update(set(string.punctuation))
stopwords_completo_es.update(set(stopwords.words('english')))

#file con nuevos stopwords
def get_stopwords_list(stop_file_path):
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return list(frozenset(stop_set))
stopwords_path_spanish = "spanish.txt"
stopwords_spanish_new = get_stopwords_list(stopwords_path_spanish)
stopwords_completo_es.update(set(stopwords_spanish_new))
extra_stop_words = {'tr', 'tnks', 'ud', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}
stopwords_completo_es.update(extra_stop_words)
#print("lista stopwords:")
#print(stopwords_completo_es)

def filtrado_stopwords(comment):
    final_string_tokens_spacy_no_stopwords = []
    for w in comment:
        if w not in stopwords_completo_es:
            final_string_tokens_spacy_no_stopwords.append(w)
    return final_string_tokens_spacy_no_stopwords

contador = 0
lista_dic_senti = []
analyzer = create_analyzer(task="sentiment", lang="es")
for x in collection.find():
    contador += 1
    print("N:", contador)
    print("Lista original:",x["tokenizado"])
    sinStopW = filtrado_stopwords(x["tokenizado"])
    print("Lista sinSW:",sinStopW)
    #print("N:",contador, "Tokenizado;",x["tokenizado"])
    oracionDeTokenizada = TreebankWordDetokenizer().detokenize(sinStopW)
    print("Lista desTokenizada:",oracionDeTokenizada)
    output = analyzer.predict(oracionDeTokenizada)
    print("Sentimiento:", output.probas)
    lista_dic_senti.append(output.probas)

print("Lista de diccionarios:")
print(lista_dic_senti)
#insertar a mongodb
cursor = collection.find({})
for comen, dic in zip(cursor,lista_dic_senti):
    collection.update_one(comen, {'$set': {'sentimiento': dic}})
"""
###FIN################

###########Ranking por sentimiento
#Ver los sentimientos de cada comentario
"""
contador = 0
for x in collection.find():
    contador += 1
    print("N:",contador, "dicSentimiento;",x["sentimiento"]["POS"])
"""

"""
contador = 0
for x in collection.find():
    contador += 1
    print("N:",contador, "dicSentimiento;",x["sentimiento"]["POS"])
"""

import statistics

#id_video = "fgdCNuGPJnw"
def promedioSentimiento(idVideo):
    #contador = 0
    valoresSenti = []
    for x in collection.find({'id_video': idVideo}):
        #contador +=1
        #print("N:",contador, "dicSentimiento;",x["sentimiento"]["POS"])
        valoresSenti.append(x["sentimiento"]["POS"])
    prom = statistics.mean(valoresSenti)
    return prom

#contador = 0
listaID = []
listaPROM = []
for x in collection.find():
    if x["id_video"] not in listaID:
        listaID.append(x["id_video"])
        #contador += 1
        listaPROM.append(promedioSentimiento(x["id_video"]))
#print("N:",contador, "len(listaID):",len(listaID))
#print(listaID)
tuplaRanking = tuple(zip(listaID, listaPROM))
tuplaRankingOrd = sorted(tuplaRanking, reverse=True)
print(tuplaRankingOrd)




#print(promedioSentimiento(id_video))
"""
cursor = collection.find({})
for comen,lista_tokenizada in zip(cursor,comentario_lematizado):
    collection.update_one(comen, {'$set': {'tokenizado': lista_tokenizada}})
"""



######Borradores####
#comentarios = pd.read_csv("u2comentariosPreProce.csv")
#print(comentarios.head())
"""
fcc_file = open('u2comentariosPreProce.json', 'r', encoding="utf8")
fcc_data = json.load(fcc_file)
#print(fcc_data)
for i in fcc_data:
    print(i["comentario"])

"""

"""
with open('u2comentariosPreProce.json', 'r', encoding="utf8") as jsonFile:
    jsonObject = json.load(jsonFile)
    jsonFile.close()

contador = 0
for i in jsonObject:
    print("id_video:", i["id_video"], ", comentario:", i["comentario"])
    contador += 1
print("cantidad de comentarios", contador)
"""