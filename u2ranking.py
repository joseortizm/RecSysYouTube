##Ranking basado en la puntuacion que darian los usuarios
from pymongo import MongoClient
MONGO_URI = "mongodb://localhost"

client = MongoClient(MONGO_URI)

dbTest = client["teststore"]
collection = dbTest["u2comentarios"]

#Calificaciones planteadas
"""
1: 0-0,15
2: 0,16-0,3
3: 0,31-0,6
4: 0,61 - 0,85
5: 0,86 - 1
"""
"""
contador = 0
for x in collection.find():
    contador += 1
    valor =x["sentimiento"]["POS"]
    if (0 < valor<=0.15):
        print("id_video",x["id_video"],"|| usuario",x["usuario"], "|| Puntuacion 1->dicSentimiento;",valor)
    elif(0.16 < valor <= 0.3):
        print("id_video",x["id_video"],"|| usuario",x["usuario"],"|| Puntuacion 2->dicSentimiento;", valor)
    elif(0.31 < valor<=0.6):
        print("id_video",x["id_video"],"|| usuario",x["usuario"],"|| Puntuacion 3->dicSentimiento;", valor)
    elif (0.61 < valor <= 0.85):
        print("id_video",x["id_video"],"|| usuario",x["usuario"],"|| Puntuacion 4->dicSentimiento;", valor)
    elif (0.86 < valor <= 1):
        print("id_video",x["id_video"],"|| usuario",x["usuario"],"|| Puntuacion 5->dicSentimiento;", valor)
"""
#####

import numpy as np
import pandas as pd
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_recommenders as tfrs
from pathlib import Path
ruta = Path("demo_puntuaciones.csv") #Se espera generar datos como demo_puntuaciones.csv donde existen 1-5 calificaciones
datos_ranking_simulado=pd.read_csv(ruta, dtype={'rating': 'int8'}, names=['userId', 'productId', 'rating', 'timestamp'], index_col=None, header=0)

class RankingModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        embedding_dimension = 32
        self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=unique_userIds, mask_token=None),
            tf.keras.layers.Embedding(len(unique_userIds) + 1, embedding_dimension)
        ])
        self.product_embeddings = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=unique_productIds, mask_token=None),
            tf.keras.layers.Embedding(len(unique_productIds) + 1, embedding_dimension)
        ])
        self.ratings = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
    def call(self, userId, productId):
        user_embeddings = self.user_embeddings(userId)
        product_embeddings = self.product_embeddings(productId)
        return self.ratings(tf.concat([user_embeddings, product_embeddings], axis=1))
class ModeloRanking(tfrs.models.Model):
    def __init__(self):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()])
    def compute_loss(self, features, training=False):
        rating_predictions = self.ranking_model(features["userId"], features["productId"])

        return self.task(labels=features["rating"], predictions=rating_predictions)


data_by_date = datos_ranking_simulado.copy()
data_by_date.timestamp = pd.to_datetime(datos_ranking_simulado.timestamp, unit="s")
data_by_date = data_by_date.sort_values(by="timestamp", ascending=False).reset_index(drop=True)

data_by_date["year"]  = data_by_date.timestamp.dt.year
rating_by_year = data_by_date.groupby(["year","month"])["rating"].count().reset_index()
rating_by_year["date"] = pd.to_datetime(rating_by_year["year"].astype("str")  +"-"+rating_by_year["month"].astype("str") +"-1")
##

cutoff_no_rat = 50    ## Solo cuente los productos que recibieron más o igual a 50
cutoff_year   = 2011  ##Solo cuenta la calificación después de 2011
recent_data   = data_by_date.loc[data_by_date["year"] > cutoff_year]
print("Cantidad de calificaciones: {:,}".format(recent_data.shape[0]) )
print("Cantidad de usuarios: {:,}".format(len(recent_data.userId.unique()) ) )
print("Cantidad de videos: {:,}".format(len(recent_data.productId.unique())  ) )
del data_by_date 
recent_prod   = recent_data.loc[recent_data.groupby("productId")["rating"].transform('count').ge(cutoff_no_rat)].reset_index(
                    drop=True).drop(["timestamp","year","month"],axis=1)
del recent_data  

##
userIds    = recent_prod.userId.unique()
productIds = recent_prod.productId.unique()
total_ratings= len(recent_prod.index)
##
ratings = tf.data.Dataset.from_tensor_slices( {"userId":tf.cast( recent_prod.userId.values  ,tf.string),
                                "productId":tf.cast( recent_prod.productId.values,tf.string),
                                "rating":tf.cast( recent_prod.rating.values  ,tf.int8,) } )
##
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take( int(total_ratings*0.8) )
test = shuffled.skip(int(total_ratings*0.8)).take(int(total_ratings*0.2))

unique_productIds = productIds
unique_userIds = userIds
##
model = ModeloRanking()
model.compile(optimizer=tf.keras.optimizers.Adagrad( learning_rate=0.1 ))
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()
model.fit(cached_train, epochs=10)
##
model.evaluate(cached_test, return_dict=True)
##
user_rand = userIds[120]
test_rating = {}
for m in test.take(10):
    test_rating[m["productId"].numpy()]=RankingModel()(tf.convert_to_tensor([user_rand]),tf.convert_to_tensor([m["productId"]]))
print("Los primeras 10 videos recomendados para el usuario {}: ".format(user_rand))
for m in sorted(test_rating, key=test_rating.get, reverse=True):
    print(m.decode())









