import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

df = pd.read_csv("data.tsv", sep="\t")
X = df.Abstract
y = df.DomainID

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, stratify=y)

tokenizer = Tokenizer(num_words=10000) 
tokenizer.fit_on_texts(X_train)
X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)


max_length = max(len(x) for x in X_train_tokens)
X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_tokens, maxlen=max_length, padding='post')

num_classes = len(set(y))
y_train_kat = to_categorical(y_train, num_classes)
y_test_kat = to_categorical(y_test, num_classes)


model = Sequential([
    Input(shape=(max_length,)),  
    Embedding(input_dim=10000, output_dim=300, input_length=max_length, trainable=True),
    Flatten(),  
    Dense(128, activation='relu'), 
    Dense(num_classes, activation='softmax') ])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit( X_train_pad, y_train_kat, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

y_pred_proba = model.predict(X_test_pad)
y_pred = np.argmax(y_pred_proba, axis=1)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

print(classification_report(y_test, y_pred))



