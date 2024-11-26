import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Load data
df = pd.read_csv("data.tsv", sep="\t")
X = df.Abstract.values
y = df.DomainID.values

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)
max_length = max(len(seq) for seq in X_sequences)
X_padded = pad_sequences(X_sequences, maxlen=max_length)

# Manual train-test split (70-30) with stratification
def stratified_split(X, y, test_size=0.3):
    classes, class_counts = np.unique(y, return_counts=True)
    train_indices = []
    test_indices = []
    for c, count in zip(classes, class_counts):
        indices = np.where(y == c)[0]
        np.random.shuffle(indices)
        split = int(count * (1 - test_size))
        train_indices.extend(indices[:split])
        test_indices.extend(indices[split:])
    return train_indices, test_indices

train_indices, test_indices = stratified_split(X_padded, y)
X_train, X_test = X_padded[train_indices], X_padded[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Convert labels to categorical
num_classes = len(np.unique(y))
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Build model
vocab_size = len(tokenizer.word_index) + 1
model = Sequential([
    Embedding(vocab_size, 300, input_length=max_length),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train_cat, epochs=10, validation_split=0.2, batch_size=32)

# Evaluate and predict
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test_cat, axis=1)

# Manual calculation of metrics
def calculate_metrics(y_true, y_pred):
    classes = np.unique(y_true)
    metrics = {}
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        metrics[c] = {'Precision': precision, 'Recall': recall, 'F1-score': f1}
    accuracy = np.mean(y_true == y_pred)
    return metrics, accuracy

class_metrics, accuracy = calculate_metrics(y_test_classes, y_pred_classes)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print("\nClass-wise metrics:")
for c, m in class_metrics.items():
    print(f"Class {c}:")
    print(f"  Precision: {m['Precision']:.4f}")
    print(f"  Recall: {m['Recall']:.4f}")
    print(f"  F1-score: {m['F1-score']:.4f}")
