# Exercise 7

This is the seventh exercise for the course [Deep Learning](https://lehre.idh.uni-koeln.de/lehrveranstaltungen/wintersemester-2024-2025/deep-learning/).

Clone this repository to your local setup. Create a branch with the name of your GitHub account and switch to it.

In the repository, you find a TSV file with science paper abstracts and associated domains ([Source](https://data.mendeley.com/datasets/9rw3vkcfy4/2))

Train a feed-forward neural network with Keras (you can choose the number of layers, neurons, activation functions and epochs) to predict the domain of an abstract.
Train the network on a train split that is 70% of the whole dataset and test on a test split that is 30%. Balance the classes in each split according to their frequencies (stratify)
Tokenize the abstracts, pad all messages according to the longest text and include a trainable embedding layer in your network. The embedding layer should have 300 dimensions.

Let the model predict on the test set and compute Accuracy, Precision, Recall and F1-score.

Commit your changes and push them to the branch with the name of your GitHub account.

Deadline for this exercise is November 28, 2024, 08:00:00 CET.
