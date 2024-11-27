import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical

df = pd.read_csv("data.tsv", sep="\t")
X = df.Abstract
y = df.DomainID

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
vocab_length = len(tokenizer.word_index) + 1
max_length = max(len(x) for x in X_train)
X_train = pad_sequences(X_train, maxlen=max_length, padding="post")
X_test = pad_sequences(X_test, maxlen=max_length, padding="post")

FFNN = Sequential(
    [
        Embedding(input_dim=vocab_length, output_dim=300),
        Flatten(),
        Dense(30, activation="relu"),
        Dense(15, activation="relu"),
        Dropout(0.3),
        Dense(len(np.unique(y)), activation="softmax"),
    ]
)

FFNN.compile(
    optimizers.SGD(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = FFNN.fit(
    X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1
)


y_pred = FFNN.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_test, y_pred))
