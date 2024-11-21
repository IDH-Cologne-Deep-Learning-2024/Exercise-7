import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
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

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

MAX_LENGTH = max(len(seq) for seq in X_train)
X_train = pad_sequences(X_train, maxlen=MAX_LENGTH, padding='post')
X_test = pad_sequences(X_test, maxlen=MAX_LENGTH, padding='post')

y_train = np.array(y_train)
y_test = np.array(y_test)

num_classes = len(np.unique(y))
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 300, input_length=MAX_LENGTH))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, verbose=1)

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1) 

print(classification_report(y_test, y_pred, digits=4))
