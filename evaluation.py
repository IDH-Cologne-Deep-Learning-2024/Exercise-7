import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("data.tsv", sep="\t")
X = df.Abstract
y = df.DomainID
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1337)

# Formatting the data
train_texts = X_train.tolist()
test_texts = X_test.tolist()
train_labels = np.array(y_train)
test_labels = np.array(y_test)

# One-Hot Encoding for the labels because multi-lable
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
vocab_size = len(tokenizer.word_index) + 1
tokenized_train_texts = tokenizer.texts_to_sequences(train_texts)
tokenized_test_texts = tokenizer.texts_to_sequences(test_texts)

# Get the overall MAX_LENGTH
max_length_train = max(len(tokenized_text) for tokenized_text in tokenized_train_texts)
max_length_test = max(len(tokenized_text) for tokenized_text in tokenized_test_texts)
MAX_LENGTH = max(max_length_train, max_length_test)

# Pad both sequence lists to MAX_LENGTH, not sure if this is necessary or even a bad idea (?)
tokenized_train_texts = pad_sequences(tokenized_train_texts, maxlen=MAX_LENGTH, padding="post")
tokenized_test_texts = pad_sequences(tokenized_test_texts, maxlen=MAX_LENGTH, padding="post")

# Model
model = Sequential()
model.add(Input(shape=(MAX_LENGTH,)))
model.add(Embedding(vocab_size, 300, input_length=MAX_LENGTH))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
# 7 is the number of different labels in the dataset, for brevity I hardcoded it
model.add(Dense(7, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="sgd")
model.fit(
    tokenized_train_texts,
    train_labels, epochs=1,
    batch_size=32,
    validation_split=0.2
)

y_pred = model.predict(tokenized_test_texts)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(test_labels, axis=1)
print(classification_report(y_test_classes, y_pred_classes))
