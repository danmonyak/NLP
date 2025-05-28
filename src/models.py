"""
models.py
=======
Author - Daniel Monyak
=======

Module containing model class definitions
Model classes do not inherit from a TensorFlow model class
    instead they hold TF models as internal attributes
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import src.wrappers as wp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class NLP_model:
    """
    Super-parent model class
    Implements a few basic methods
    """
    def getWeights(self):
        """
        Retrieve the current weights of the model.

        Returns:
            list: A list of NumPy arrays representing the model's weights.
        """
        return self.model.get_weights()
    
    def fit(self, X, Y, verbose=2, epochs=80):
        """
        Train the model on the given input data and labels.

        Args:
            X (array-like): Input features for training.
            Y (array-like): Target labels for training.
            verbose (int, optional): Verbosity level of training output. Defaults to 2.
            epochs (int, optional): Number of training epochs. Defaults to 80.

        Also updates self.weights with the trained model's weights.
        """
        self.model.fit(X, Y, verbose=verbose, epochs=epochs)
        self.weights = self.model.get_weights()
    
    def save(self, model_path):
        """
        Save the trained model to the specified file path.

        Args:
            model_path (str): File path where the model should be saved.
        """
        self.model.save(model_path)


class Embedding_model(NLP_model):
    """
    Parent class for all models that learn word embeddings
    Child on NLP_model super-parent class
    """
    def __init__(self, embedding_dim=None, model_path=None, text_object=None):
        """
        Initialize an Embedding_model instance.

        Args:
            embedding_dim (int, optional): Dimensionality of the embedding vectors. 
                Required if creating a new model.
            model_path (str, optional): Path to a pre-trained model. If provided, 
                the model will be loaded from this path instead of being created.
            text_object (object, optional): A text processing object that must contain
                unique_words, word_counts, and word_to_idx. Required when creating a new model.

        Attributes:
            self.model: The Keras model instance.
            self.weights: List of model weights.
            self.n_unique_words (int): Number of unique words in the vocabulary.
            self.embedding_dim (int): Dimensionality of the embedding vectors.
        """
        self.text_object = text_object
        if model_path is None:
            self.n_unique_words = len(self.text_object.unique_words)
            self.embedding_dim = embedding_dim
            self.createCompileModel()
        else:
            self.model = wp.load_model(model_path)
            self.weights = self.model.get_weights()
            retrieved_info = self.retrieveInfo()
            self.n_unique_words = retrieved_info['n_unique_words']
            self.embedding_dim = retrieved_info['embedding_dim']
        
    def getEmbeddings(self):
        """
        Retrieve the learned embedding matrix from the model's weights.

        Returns:
            numpy.ndarray: The embedding weight matrix (typically the first weight matrix).
        """
        return self.weights[0]

    def plotWords(self, n=20, word_sample=None):
        """
        Visualize word embeddings in 2D space using PCA for dimensionality reduction.

        Args:
            n (int, optional): Number of words to plot. Ignored if `word_sample` is provided. Defaults to 20.
            word_sample (list, optional): Specific list of words to plot. If None, a random sample
                of frequent words is selected.

        Notes:
            - StandardScaler is applied to normalize embeddings before PCA.
            - Words are plotted as red text at their projected 2D coordinates.
        """
        embeddings_scaled = StandardScaler().fit_transform(self.getEmbeddings())
        embeddings_pca = PCA(n_components=2).fit_transform(embeddings_scaled)

        fig, ax = plt.subplots()

        if word_sample is None:
            word_sample = np.random.default_rng().choice(
                self.text_object.word_counts.index[self.text_object.word_counts > 50].values,
                size=n, replace=False)
        
        xmin = float('Inf')
        ymin = float('Inf')
        xmax = float('-Inf')
        ymax = float('-Inf')

        for word in word_sample:
            word_idx = self.text_object.word_to_idx[word]
            x, y = embeddings_pca[word_idx]
            plt.text(x, y, word, color='red')

            xmin = min(xmin, x)
            ymin = min(ymin, y)
            xmax = max(xmax, x)
            ymax = max(ymax, y)

        xrg = xmax - xmin
        yrg = ymax - ymin
        ext = 0.2
        ax.set_xlim(xmin - xrg * ext, xmax + xrg * ext)
        ax.set_ylim(ymin - yrg * ext, ymax + yrg * ext)
        plt.show()

class CBOW_model(Embedding_model):
    """
    Class that implements the CBOW model for learning word embeddings
    Child class of Embedding_model
    """
    def retrieveInfo(self):
        """
        Extract model configuration details from the trained weights.

        Returns:
            dict: A dictionary containing:
                - 'n_unique_words' (int): Number of unique words in the vocabulary,
                  derived from the input embedding matrix shape.
                - 'embedding_dim' (int): Dimensionality of the embedding vectors,
                  inferred from the shape of the hidden layer weights.
        """
        return {
            'n_unique_words': self.weights[0].shape[0],
            'embedding_dim': self.weights[2].shape[0]
        }
    
    def createCompileModel(self):
        """
        Create and compile a CBOW (Continuous Bag-of-Words) model using Keras.

        Architecture:
            - InputLayer with size equal to the vocabulary size (one-hot encoding).
            - Dense hidden layer with ReLU activation (embedding layer).
            - Output layer with softmax activation to predict the target word.

        Compiles the model with:
            - Adam optimizer.
            - Sparse categorical crossentropy loss.
            - Accuracy as a metric.
        """
        self.model = keras.models.Sequential([
            keras.layers.InputLayer(self.n_unique_words),
            keras.layers.Dense(self.embedding_dim, activation="relu"),
            keras.layers.Dense(self.n_unique_words, activation='softmax')
        ])
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

class PredictionModel(NLP_model):
    """
    Model that predicts the next word given the previous k words
    Uses word embeddings of k words as features
    Supports random sentence generation
    Inherits from NLP_model
    """
    def __init__(self, k, PN_obj=None, architecture=None, model_path=None):
        """
        Initialize a PredictionModel instance.

        Args:
            k (int): Number of context words used for prediction (e.g., in a k-gram model).
            PN_obj (object, optional): A pre-trained model or object containing word embeddings,
                vocabulary size, and other metadata.
            architecture (list of int, optional): List specifying the number of neurons in each hidden layer.
                Required if creating a new model.
            model_path (str, optional): Path to a saved model. If provided, the model will be loaded from disk
                and PN_obj/architecture will be ignored.

        Attributes:
            self.k (int): Number of context words.
            self.embedding_dim (int): Dimensionality of the embeddings used as input.
            self.PN_n_unique_words (int): Number of unique words in the PN_obj vocabulary.
            self.architecture (list): Architecture of the neural network.
            self.model (tf.keras.Model): The compiled prediction model.
        """
        self.k = k
        self.PN_obj = PN_obj
        if model_path is None:
            self.embedding_dim = self.PN_obj.embedding_dim
            self.PN_n_unique_words = self.PN_obj.unique_words_PN.shape[0]
            self.architecture = architecture
            self.createCompileModel()
        else:
            self.model = wp.load_model(model_path)
            self.weights = self.model.get_weights()

            retrieved_info = self.retrieveInfo()
            self.embedding_dim = retrieved_info['embedding_dim']
            self.PN_n_unique_words = retrieved_info['PN_n_unique_words']
            self.architecture = retrieved_info['architecture']

    def retrieveInfo(self):
        """
        Extract model configuration from the weights of a loaded model.

        Returns:
            dict: A dictionary containing:
                - 'embedding_dim' (float): Inferred embedding dimensionality, based on input layer.
                - 'PN_n_unique_words' (int): Output vocabulary size, inferred from final layer shape.
                - 'architecture' (list): Number of units in each hidden layer, inferred from weight shapes.
        """
        return {
            'embedding_dim': self.weights[0].shape[0] / self.k,
            'PN_n_unique_words': self.weights[-1].shape[0],
            'architecture': [wt.shape[0] for wt in self.weights[1:-1:2]]
        }

    def createCompileModel(self):
        """
        Construct and compile a feedforward neural network for word prediction.

        Architecture:
            - Input layer expects a vector of length (embedding_dim × k).
            - Several hidden layers as specified in self.architecture, each with ReLU activation.
            - Output layer with softmax activation to predict the next word.

        Compiles the model using:
            - Adam optimizer.
            - Sparse categorical crossentropy loss.
            - Accuracy as the evaluation metric.
        """
        layer_list = []
        layer_list.append(keras.layers.InputLayer(self.embedding_dim * self.k))
        for n_nodes in self.architecture:
            layer_list.append(keras.layers.Dense(n_nodes, activation="relu"))
        layer_list.append(keras.layers.Dense(self.PN_n_unique_words, activation='softmax'))

        self.model = tf.keras.models.Sequential(layer_list)

        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def genSentence(self):
        """
        Generate a sentence using the trained prediction model and PN_obj embeddings.

        Returns:
            str: A generated sentence with up to 30 words, ending with a period if predicted.

        Process:
            - Starts with k placeholder indices (-1) to represent context.
            - Repeatedly predicts the next word based on current context embeddings.
            - Samples the next word from the model’s probability output.
            - Updates the context window and continues until a period is predicted or the
              sentence reaches 30 words.

        Notes:
            - Uses multinomial sampling to add randomness to the generation.
            - Uses PN_obj.getPN_Embedding() to retrieve word embeddings by index.
        """
        gen = np.random.default_rng()

        working_sentence_idxs = [-1 for _ in range(self.k)]
        built_sentence = []

        st_len = 0
        while st_len < 30:
            # Get concatenated embeddings for current context
            working_sentence_embeddings = np.concatenate([
                self.PN_obj.getPN_Embedding(idx) for idx in working_sentence_idxs
            ])

            # Predict probabilities for the next word
            next_word_probs = np.reshape(
                self.model.predict(np.reshape(working_sentence_embeddings, [1, -1]), verbose=0),
                [-1,]
            )

            # Sample next word index using predicted probabilities
            next_word_idx = gen.choice(range(next_word_probs.shape[0]), size=1, p=next_word_probs)[0]
            next_word = self.PN_obj.unique_words_PN[next_word_idx]

            built_sentence.append(next_word)
            st_len += 1

            if next_word == '.':
                break

            # Slide the context window
            working_sentence_idxs = working_sentence_idxs[1:] + [next_word_idx]

        line = ' '.join(built_sentence)
        return line.replace(' .', '.').capitalize()
