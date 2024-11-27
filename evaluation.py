import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# ffnn keras split 70/30 train/test, balance via stratify, tokenize abstracts, pad msg, 300d trainable embedding layer in network
# predict in test set, compute A P R F1;
# mentioned alternatives to use when working with lots of datra: sub-sample, more layers, smaller learning rate

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

df = pd.read_csv("data.tsv", sep="\t")
df = df.sample(frac= 0.5, random_state=13)
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
model.add(Dense(40, activation="relu")) #alt new neurons
model.add(Dense(7, activation="softmax")) #alt new layers -- also rewrite possible to incorporate input classes
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]) #alt sgd learning rate 
model.fit(tokenized_texts_train_pad, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=2)

y_pred = np.argmax(model.predict(tokenized_texts_test_pad), axis=1)
print(classification_report(y_test, y_pred))