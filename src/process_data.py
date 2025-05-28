"""
process_data.py
=======
Author - Daniel Monyak
=======

Module with functions to help process text data
"""

"""
Retrieve text data and prepare into sequence of words
"""

import os
import re
import numpy as np
import pandas as pd
def getSentences(filepath, max_lines=int(1e5)):
    """
    Read a text file and extract cleaned sentences as lists of words.

    This function processes a text file line by line, performing basic text cleaning,
    punctuation handling, and sentence segmentation. It returns a list of tokenized sentences,
    where each sentence is a list of lowercase words.

    Args:
        filepath (str): Path to the input text file.
        max_lines (int, optional): Maximum number of lines to read from the file.
            Defaults to 1e5.

    Returns:
        list of list of str: A list of sentences, where each sentence is a list of cleaned, lowercase words.

    Processing steps:
        - Reads the file line by line up to `max_lines`.
        - Strips leading/trailing whitespace.
        - Treats blank lines or sentence-ending punctuation (., ?, !) as sentence boundaries.
        - Removes all punctuation except periods, question marks, and exclamation points,
          which are normalized to periods.
        - Normalizes spacing, tabs, and converts words to lowercase.
        - Skips empty lines and sentences.
        - Uses '.' to mark sentence boundaries explicitly.

    Example:
        Input file:
            Hello world!
            This is a test.

        Output:
            [['hello', 'world'], ['this', 'is', 'a', 'test']]
    """

    sentence_list = []
    sentence = []
    line_num = 0

    with open(filepath, 'r') as f:
        while True:
            line = f.readline()
            line_num += 1

            if (line == '') or (line_num > max_lines):
                break

            line = line.strip()

            if line == '':
                if len(sentence) > 0:
                    sentence_list.append(sentence)
                    sentence = []
                continue

            # Normalize punctuation
            line = re.sub('[^a-zA-Z \.\?\!]', '', line)
            line = re.sub('\.', ' .', line)
            line = re.sub('\?', ' .', line)
            line = re.sub('\!', ' .', line)
            line = re.sub(' +', ' ', line)
            line = re.sub('\t', ' ', line)
            line = line.strip()

            fields = line.split(' ')

            for fld in fields:
                if fld in ['.', '?', '!']:
                    if len(sentence) > 0:
                        sentence_list.append(sentence)
                        sentence = []
                else:
                    sentence.append(fld.lower())

    return sentence_list

def getUniqueWords(sentence_list):
    """
    Extract a sorted list of unique words from a list of tokenized sentences.

    Args:
        sentence_list (list of list of str): A list where each element is a tokenized sentence,
            represented as a list of lowercase words.

    Returns:
        np.ndarray: A sorted NumPy array of unique words present across all sentences.

    Example:
        Input:
            [['the', 'cat', 'sat'], ['on', 'the', 'mat']]
        Output:
            array(['cat', 'mat', 'on', 'sat', 'the'], dtype='<U3')

    Notes:
        - Uses NumPy for fast concatenation and uniqueness computation.
        - Assumes all input words are already preprocessed (lowercased, cleaned).
    """
    unique_words = sorted(set(word for sentence in sentence_list for word in sentence))

    return unique_words

def returnTrainData(sentence_list, unique_words):
    """
    Generate training data for a CBOW-like model using a sliding window of 3 words.

    Constructs (X, Y) training pairs from a list of tokenized sentences using
    a context–target–context pattern. Specifically, the first and third words (context)
    are used to predict the middle word (target).

    Args:
        sentence_list (list of list of str): Tokenized sentences (each a list of words).
        unique_words (np.ndarray): Sorted list of unique words in the corpus.

    Returns:
        X (np.ndarray): Input features of shape (num_samples, vocab_size).
            Each row is a one-hot vector combining the two context words.
        Y (np.ndarray): Target word indices of shape (num_samples, 1).
        word_to_idx (dict): Dictionary mapping each word to its index in the vocabulary.

    Example:
        For the sentence: ['the', 'cat', 'sat'], one training pair would be:
            X = one-hot('the') + one-hot('sat')
            Y = index of 'cat'

    Notes:
        - Uses one-hot encoding to represent words.
        - Only considers trigrams (3 consecutive words).
        - Skips sentences shorter than 3 words.
        - Assumes all words in sentence_list exist in unique_words.
    """
    # Build a word-to-index mapping
    word_to_idx = {}
    for word_idx, word in enumerate(unique_words):
        word_to_idx[word] = word_idx

    n_features = len(unique_words)

    # X and Y lists for storing samples
    X_list = []
    Y_list = []

    # Iterate over sentences
    for sentence in sentence_list:
        idx_in_sentence = 0

        # Slide a window of 3 words across each sentence
        while (idx_in_sentence + 2) < len(sentence):
            X_sample = np.zeros(n_features)

            word0, word1, word2 = sentence[idx_in_sentence : (idx_in_sentence + 3)]

            # Use context words (word0 and word2) as input
            X_sample[word_to_idx[word0]] = 1
            X_sample[word_to_idx[word2]] = 1
            X_list.append(X_sample)

            # Use center word (word1) as label
            Y_list.append(word_to_idx[word1])

            idx_in_sentence += 1

    # Convert lists to NumPy arrays
    X = np.stack(X_list)
    Y = np.reshape(np.stack(Y_list), [-1, 1])

    return X, Y, word_to_idx

class TextObject:
    """
    A container class for processing and storing text data and related structures
    for training word embedding models like CBOW.

    This class takes a list of tokenized sentences and automatically extracts:
      - A vocabulary of unique words
      - One-hot encoded training data for embedding models
      - Word-to-index mappings
      - Word frequency counts

    Attributes:
        sentence_list (list of list of str): The original list of tokenized sentences.
        unique_words (np.ndarray): Sorted array of unique words in the corpus.
        embedding_X (np.ndarray): One-hot encoded input features for CBOW-style training.
        embedding_Y (np.ndarray): Target word indices corresponding to embedding_X.
        word_to_idx (dict): Dictionary mapping words to their index positions.
        all_words (np.ndarray): Flattened array of all words in the corpus.
        word_counts (pd.Series): Word frequency counts, sorted in descending order.

    Args:
        sentence_list (list of list of str): A list of tokenized sentences.

    Example:
        text_obj = TextObject([['the', 'cat', 'sat'], ['on', 'the', 'mat']])
        print(text_obj.unique_words)
        print(text_obj.word_to_idx)
    """
    def __init__(self, sentence_list):
        self.sentence_list = sentence_list
        self.unique_words = getUniqueWords(sentence_list)
        self.embedding_X, self.embedding_Y, self.word_to_idx = returnTrainData(self.sentence_list, self.unique_words)

        self.all_words = np.concatenate(sentence_list)
        self.word_counts = pd.Series(self.all_words).value_counts().sort_values(ascending=False)


class PredictNext:
    """
    A utility class for generating training data and embeddings for predicting the next word
    in a sentence using word embeddings and a sliding context window of length `k`.

    This class supports context-based prediction using a pre-trained embedding model
    (e.g., CBOW), and prepares features suitable for training a next-word prediction model.

    Attributes:
        embedding_model (Embedding_model): A trained embedding model that provides word vectors.
        text_object (TextObject): An instance of TextObject containing sentence and word data.
        k (int): Number of previous words (context length) used to predict the next word.
        embedding_dim (int): Dimensionality of the word embeddings.
        unique_words_PN (np.ndarray): Unique words plus a terminal token ('.') for sentence ends.
        period_idx (int): Index of the terminal period token in the extended vocabulary.

    Methods:
        getPN_Embedding(idx):
            Returns the embedding vector for a word given its index.
            If idx == -1 (used for padding), returns a zero vector.

        returnTrainData():
            Constructs training data for predicting the next word given the `k` previous words.
            Returns:
                PN_X (np.ndarray): Input features of shape (num_samples, k * embedding_dim).
                PN_Y (np.ndarray): Target word indices of shape (num_samples, 1).
    """

    def __init__(self, embedding_model, text_object, k):
        self.embedding_model = embedding_model
        self.text_object = text_object
        self.k = k

        self.embedding_dim = self.embedding_model.embedding_dim
        self.unique_words_PN = np.concatenate([self.text_object.unique_words, ['.']])
        self.period_idx = self.unique_words_PN.shape[0] - 1

    def getPN_Embedding(self, idx):
        """
        Get the embedding for a word index. If index is -1 (used for padding), return a zero vector.

        Args:
            idx (int): Index of the word in the vocabulary or -1 for padding.

        Returns:
            np.ndarray: The word's embedding or a zero vector if padding.
        """
        if idx == -1:
            return np.zeros(self.embedding_dim)
        return self.embedding_model.getEmbeddings()[idx]

    def returnTrainData(self):
        """
        Generate training data for next-word prediction using `k` previous embeddings.

        For each sentence in the corpus, it constructs (X, Y) pairs where:
        - X is the concatenation of embeddings for the previous `k` words.
        - Y is the index of the next word in the `unique_words_PN` vocabulary.

        Sentences are padded on the left with -1 indices, and a period token is appended
        at the end to mark sentence termination.

        Returns:
            tuple:
                PN_X (np.ndarray): Feature matrix of shape (num_samples, k * embedding_dim).
                PN_Y (np.ndarray): Target word indices of shape (num_samples, 1).
        """
        PN_X_list = []
        PN_Y_list = []

        for st_idx, sentence in enumerate(self.text_object.sentence_list):
            if len(sentence) < 4:
                continue

            sentence_as_idxs = [-1 for _ in range(self.k)] + \
                               [self.text_object.word_to_idx[word] for word in sentence] + \
                               [self.period_idx]

            idx_in_sentence = 0
            while (idx_in_sentence + self.k) < len(sentence_as_idxs):
                idxs_cur = sentence_as_idxs[idx_in_sentence : (idx_in_sentence + self.k)]
                X_cur = np.concatenate([self.getPN_Embedding(idx) for idx in idxs_cur])
                PN_X_list.append(X_cur)

                PN_Y_list.append(sentence_as_idxs[idx_in_sentence + self.k])

                idx_in_sentence += 1

        PN_X = np.stack(PN_X_list)
        PN_Y = np.reshape(np.stack(PN_Y_list), [-1, 1])

        return PN_X, PN_Y
