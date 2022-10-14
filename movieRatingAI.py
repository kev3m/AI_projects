# Classificação de texto com avaliações de filmes
import tensorflow as tf
from tensorflow import keras
import numpy as np

#consulta um dicionario que contem inteiros mapeados em strings
def decode_review(text):
    return ' '.join([reverse_word_dict.get(i,'?') for i in text])

#Baixando base de dados imdb
imdb = keras.datasets.imdb

(train_data,train_labels), (test_data,test_labels) = imdb.load_data(num_words=10000) #num_words=10000 mantém as 10000 palavras mais frequentes.
 
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
# print(train_data[0]) #Palavras na avaliação [0] -> Cada inteiro representa uma palavra.
print(len(train_data[0])) #Número de palavras na avaliação [0]


#---------- Convertendo inteiros em palavras ----------------
word_dict = imdb.get_word_index()


# for movie in word_dict.items():
#     print(movie[0])

word_dict = {i:(j+3) for i,j in word_dict.items()}
word_dict["<PAD>"] = 0
word_dict["<START>"] = 1
word_dict["<UNK>"] = 2 #desconhecido
word_dict["<UNUSED>"] = 3

reverse_word_dict = dict([(value, key) for (key, value) in word_dict.items()])

print(decode_review(train_data[0]))
