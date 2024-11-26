import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

df = pd.read_csv("data.tsv", sep="\t")
X = df['Abstract']
y = df['DomainID']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y)  

tokenizer = Tokenizer(num_words=20000) 
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)

max_len = max(len(seq) for seq in X_seq)  
X_padded = pad_sequences(X_seq, maxlen=max_len, padding='post')

X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.3, stratify=np.argmax(y, axis=1), random_state=42)

vocab_size = len(tokenizer.word_index) + 1  
embedding_dim = 300  

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len, trainable=True),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax') 
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_))

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
