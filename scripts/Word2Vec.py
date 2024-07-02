#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import logging
import requests
import gdown
from collections import defaultdict

logger = logging.getLogger(__name__)

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Define constants
VECTOR_SIZE = 300
VOCAB_SIZE = 3000000  # Number of words in Google News pre-trained model

# Download the pre-trained Google News word2vec model if not already downloaded
def download_pretrained_model():
    url = "https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM"
    output = "GoogleNews-vectors-negative300.bin.gz"
    if not os.path.isfile(output):
        logger.info(f"Downloading {url}...")
        gdown.download(url, output, quiet=False)
    return output

# Load pre-trained word2vec model
def load_word2vec_model(filepath):
    logger.info("Loading pre-trained word2vec model...")
    word_vectors = {}
    with open(filepath, 'rb') as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * vector_size
        for _ in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    word = b''.join(word).decode('utf-8')
                    break
                if ch != b'\n':
                    word.append(ch)
            word_vectors[word] = np.frombuffer(f.read(binary_len), dtype='float32')
    return word_vectors

# Implement the skip-gram model
def skipgram_model(word_vectors, context_words, target_word, vector_size=VECTOR_SIZE, learning_rate=0.01):
    # Initialize weight matrices
    W1 = np.random.rand(len(word_vectors), vector_size)
    W2 = np.random.rand(vector_size, len(word_vectors))
    
    target_index = word_vectors.get(target_word)
    if target_index is None:
        return None
    
    context_indices = [word_vectors.get(word) for word in context_words if word in word_vectors]
    if not context_indices:
        return None
    
    for context_index in context_indices:
        h = W1[target_index, :]
        u = np.dot(W2.T, h)
        y_pred = softmax(u)
        e = np.zeros_like(y_pred)
        e[context_index] = 1 - y_pred[context_index]
        W1[target_index, :] += learning_rate * np.dot(W2, e)
        W2[:, context_index] += learning_rate * np.dot(h[:, np.newaxis], e[np.newaxis, :])
    
    return W1, W2

# Implement the CBOW model
def cbow_model(word_vectors, context_words, target_word, vector_size=VECTOR_SIZE, learning_rate=0.01):
    # Initialize weight matrices
    W1 = np.random.rand(len(word_vectors), vector_size)
    W2 = np.random.rand(vector_size, len(word_vectors))
    
    target_index = word_vectors.get(target_word)
    if target_index is None:
        return None
    
    context_indices = [word_vectors.get(word) for word in context_words if word in word_vectors]
    if not context_indices:
        return None
    
    h = np.mean(W1[context_indices, :], axis=0)
    u = np.dot(W2.T, h)
    y_pred = softmax(u)
    e = np.zeros_like(y_pred)
    e[target_index] = 1 - y_pred[target_index]
    W1[context_indices, :] += learning_rate * np.dot(W2, e) / len(context_indices)
    W2[:, target_index] += learning_rate * np.dot(h[:, np.newaxis], e[np.newaxis, :])
    
    return W1, W2

# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Main function
def main():
    model_file = download_pretrained_model()
    word_vectors = load_word2vec_model(model_file)
    
    context_words = ['king', 'man']
    target_word = 'queen'
    
    W1_sg, W2_sg = skipgram_model(word_vectors, context_words, target_word)
    logger.info("Skip-gram model trained.")
    
    W1_cbow, W2_cbow = cbow_model(word_vectors, context_words, target_word)
    logger.info("CBOW model trained.")
    
    logger.info("Most similar words to 'queen' in Skip-gram model:")
    similar_words_sg = np.dot(W2_sg.T, W1_sg[word_vectors[target_word], :])
    print(similar_words_sg.argsort()[-10:][::-1])
    
    logger.info("Most similar words to 'queen' in CBOW model:")
    similar_words_cbow = np.dot(W2_cbow.T, W1_cbow[word_vectors[target_word], :])
    print(similar_words_cbow.argsort()[-10:][::-1])

if __name__ == "__main__":
    main()
