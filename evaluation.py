import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ffnn keras split 70/30 train/test, balance via stratify, tokenize abstracts, pad msg, 300d trainable embedding layer in network
# predict in test set, compute A P R F1
# tbc

df = pd.read_csv("data.tsv", sep="\t")
X = df.Abstract
y = df.DomainID
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=23)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1  # +1 to account for the padding token
tokenized_texts_train = tokenizer.texts_to_sequences(X_train)
tokenized_texts_test = tokenizer.texts_to_sequences(X_test)
MAX_LENGTH = max(len(seq) for seq in tokenized_texts_train)
tokenized_texts_train_pad = pad_sequences(tokenized_texts_train, maxlen=MAX_LENGTH, padding="post")
tokenized_texts_test_pad = pad_sequences(tokenized_texts_test, maxlen=MAX_LENGTH, padding="post")

model = Sequential()
model.add(Input(shape=(MAX_LENGTH,)))
model.add(Embedding(vocab_size, 300, input_length=MAX_LENGTH))
model.add(Flatten())
model.add(Dense(100, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
model.fit(tokenized_texts_train, y_train, epochs=50, verbose=1)
embeddings = model.layers[0].get_weights()
print(embeddings)

y_pred = model.predict()
print(classification_report(y_test, y_pred))
