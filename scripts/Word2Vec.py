import numpy as np
import os
import logging
import requests
import gdown

logger = logging.getLogger(__name__)

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Word2VecModel:
    VECTOR_SIZE = 300
    VOCAB_SIZE = 3000000  # Number of words in Google News pre-trained model

    def __init__(self):
        self.word_vectors = {}
        self.W1_sg = None
        self.W2_sg = None
        self.W1_cbow = None
        self.W2_cbow = None

    # Download the pre-trained Google News word2vec model if not already downloaded
    def download_pretrained_model():
        url = "https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM"
        output = "GoogleNews-vectors-negative300.bin.gz"
        if not os.path.isfile(output):
            logger.info(f"Downloading {url}...")
            gdown.download(url, output, quiet=False)
        return output

    def load_word2vec_model(self, filepath):
        logger.info("Loading pre-trained word2vec model...")
        self.word_vectors = {}
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
                self.word_vectors[word] = np.frombuffer(f.read(binary_len), dtype='float32')

    def skipgram_model(self, context_words, target_word, vector_size=VECTOR_SIZE, learning_rate=0.01):
        # Initialize weight matrices with small random values
        W1 = np.random.rand(len(self.word_vectors), vector_size)
        W2 = np.random.rand(vector_size, len(self.word_vectors))
        
        target_index = self.word_vectors.get(target_word)
        if target_index is None:
            return None # Return none if target word is not in vocabulary
        
        context_indices = [self.word_vectors.get(word) for word in context_words if word in self.word_vectors]
        if not context_indices:
            return None # Return none if no valid context words are found
        
        # Training loop for each context word
        for context_index in context_indices:
            # Forward pass
            h = W1[target_index, :]  # Get target word vector from W1
            u = np.dot(W2.T, h)  # Compute scores for all words
            y_pred = self.softmax(u)  # Apply softmax to get probabilities
            
            # Compute the error
            e = np.copy(y_pred)  # Copy predicted probabilities
            e[context_index] -= 1  # Subtract 1 from the true context word probability
            
            # Backpropagation and weight updates
            W1[target_index, :] += learning_rate * np.dot(W2, e)  # Update W1 for target word
            W2[:, context_index] += learning_rate * np.dot(h[:, np.newaxis], e[np.newaxis, :])  # Update W2 for context word
        
        # Store the updated weight matrices
        self.W1_sg = W1
        self.W2_sg = W2

    def cbow_model(self, context_words, target_word, vector_size=VECTOR_SIZE, learning_rate=0.01):
        W1 = np.random.rand(len(self.word_vectors), vector_size)
        W2 = np.random.rand(vector_size, len(self.word_vectors))
        
        target_index = self.word_vectors.get(target_word)
        if target_index is None:
            return None
        
        context_indices = [self.word_vectors.get(word) for word in context_words if word in self.word_vectors]
        if not context_indices:
            return None
        
        h = np.mean(W1[context_indices, :], axis=0)
        u = np.dot(W2.T, h)
        y_pred = self.softmax(u)
        e = np.zeros_like(y_pred)
        e[target_index] = 1 - y_pred[target_index]
        W1[context_indices, :] += learning_rate * np.dot(W2, e) / len(context_indices)
        W2[:, target_index] += learning_rate * np.dot(h[:, np.newaxis], e[np.newaxis, :])
        
        self.W1_cbow = W1
        self.W2_cbow = W2

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def most_similar_skipgram(self, word, topn=10):
        if word not in self.word_vectors:
            return None
        if self.W1_sg is None or self.W2_sg is None:
            return None
        word_index = self.word_vectors[word]
        similar_words = np.dot(self.W2_sg.T, self.W1_sg[word_index, :])
        top_indices = np.argsort(similar_words)[-topn:][::-1]
        return [(self.word_vectors.index_to_key[index], similar_words[index]) for index in top_indices]

    def most_similar_cbow(self, word, topn=10):
        if word not in self.word_vectors:
            return None
        if self.W1_cbow is None or self.W2_cbow is None:
            return None
        word_index = self.word_vectors[word]
        similar_words = np.dot(self.W2_cbow.T, self.W1_cbow[word_index, :])
        top_indices = np.argsort(similar_words)[-topn:][::-1]
        return [(self.word_vectors.index_to_key[index], similar_words[index]) for index in top_indices]

# Usage example
if __name__ == "__main__":
    model = Word2VecModel()
    model_file = model.download_pretrained_model()
    model.load_word2vec_model(model_file)
    
    context_words = ['king', 'man']
    target_word = 'queen'
    
    model.skipgram_model(context_words, target_word)
    logger.info("Skip-gram model trained.")
    
    model.cbow_model(context_words, target_word)
    logger.info("CBOW model trained.")
    
    logger.info("Most similar words to 'queen' in Skip-gram model:")
    print(model.most_similar_skipgram('queen'))
    
    logger.info("Most similar words to 'queen' in CBOW model:")
    print(model.most_similar_cbow('queen'))
