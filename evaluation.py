import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input,Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("data.tsv", sep="\t")
X = df.Abstract
y = df.DomainID
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.03, train_size=0.07)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
maxlen = max(len(x) for x in X_train)
X_train = pad_sequences(X_train, maxlen=maxlen, padding="post")
X_test = pad_sequences(X_test, maxlen=maxlen, padding="post")

model = Sequential()
model.add(Input(shape=(maxlen,)))
model.add(Embedding(vocab_size, 300))
model.add(Flatten())
model.add(Dense(60, activation="sigmoid"))
model.add(Dense(30, activation="sigmoid"))
model.add(Dense(10, activation="sigmoid"))
model.add(Dense(len(np.unique(y)), activation="softmax"))
optimizer = optimizers.SGD(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy")
model.fit(X_train, y_train, epochs=10)
model.summary()

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred))
