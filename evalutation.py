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

y_pred = model.predict()
print(classification_report(y_test, y_pred))


