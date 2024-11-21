import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("data.tsv", sep="\t")
X = df.Abstract
y = df.DomainID
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
vocabSize = len(tokenizer.word_index)+1
X_train = tokenizer.texts_to_sequences(X_train)

MAX_LENGTH = max(len(X_train) for X_train in X_train)
X_train = pad_sequences(X_train, maxlen=MAX_LENGTH, padding='post')

model = Sequential()
model.add(Embedding(vocabSize, 300, input_length=MAX_LENGTH))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd')
model.fit(X_train, y_train, epochs=10, verbose=1)

#y_pred = model.predict()
#print(classification_report(y_test, y_pred))
