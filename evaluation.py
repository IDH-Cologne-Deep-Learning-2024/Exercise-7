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
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = Sequential()
model.add(Input(shape=(2,)))
model.add(Dense(100, activation="sigmoid"))
model.add(Dense(100, activation="sigmoid"))
model.add(Dense(100, activation="sigmoid"))
model.add(Dense(100, activation="sigmoid"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="sgd")

model.fit(X_train,len(y_train),epochs=10,verbose=1)

y_pred = model.predict(y_test)
print(classification_report(y_test, y_pred))
