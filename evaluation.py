import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("data.tsv", sep="\t")
X = df.Abstract
y = df.DomainID
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1
tokenized_x_train = tokenizer.texts_to_sequences(X_train)
tokenized_x_test = tokenizer.texts_to_sequences(X_test)
max_length = max(len(sequence) for sequence in tokenized_x_train)
x_train_padded = pad_sequences(tokenized_x_train, maxlen = max_length, padding = "post")
x_test_padded = pad_sequences(tokenized_x_test, maxlen = max_length, padding = "post")

FFNN = Sequential()
FFNN.add(Input(shape=(max_length,)))
FFNN.add(Embedding(vocab_size, 187, input_length = max_length))
FFNN.add(Flatten())
FFNN.add(Dense(128, activation = "sigmoid"))
#FFNN.add(Dense(64, activation = "sigmoid"))
FFNN.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
x_train=tf.convert_to_tensor(x_train_padded) 
FFNN.fit(tokenized_x_train, y_train, epochs=69, verbose=1)

y_pred = FFNN.predict(x_test_padded)
print(classification_report(y_test, y_pred))