# Classificação de texto com avaliações de filmes
import tensorflow as tf
from tensorflow import keras
import numpy as np

print(tf.__version__)

#Baixando base de dados imdb
imdb = keras.datasets.imdb

(train_data,train_labels), (test_data,test_labels) = imdb.load_data(num_words=10000) #num_words=10000 mantém as 10000 palavras mais frequentes.
 