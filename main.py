"""
================================================================================
    TEXT GENERATION PIPELINE: CBOW + PREDICT-NEXT MODEL
================================================================================

This script performs end-to-end training and/or loading of two models:
    1. A CBOW embedding model to learn word vectors from text.
    2. A "PredictNext" neural network model to generate new sentences based on
       learned word embeddings and a fixed context window of size `k`.

The pipeline includes:
    - Reading and preprocessing raw text data.
    - Creating training data for word embeddings and prediction tasks.
    - Fitting or loading models from disk.
    - Generating new sentences based on trained models.

Parameters:
    - filename: Name of the text file in the /data directory.
    - embedding_dim: Dimensionality of word embeddings.
    - load_embedding_model: If True, loads the embedding model from disk.
    - k: Context window size for prediction.
    - PN_architecture: List defining the architecture of the prediction model.
    - load_PN_model: If True, loads the prediction model from disk.

Outputs:
    - Trained models are saved under /saved_models/{filename_prefix}/
    - Generated sentences are saved in /outputs/{filename_prefix}/

Dependencies:
    - numpy
    - pandas
    - tensorflow (via models)
    - src.process_data (data preprocessing)
    - src.models (CBOW and PredictNext model definitions)
"""

import numpy as np
import pandas as pd
import os
import src.process_data as proc
import src.models as models

##################################################################
######################## Parameters  #############################
filename = 'mockingbird.txt'
embedding_dim = 100
load_embedding_model = True
k = 6
PN_architecture = [300, 500]
load_PN_model = False
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################

# Create output directories
textname = filename.split('.txt')[0]
os.makedirs(os.path.join('saved_models', textname), exist_ok=True)
os.makedirs(os.path.join('outputs', textname), exist_ok=True)

# Process text
sentence_list = proc.getSentences(
        os.path.join('data', filename)
    )
training_text = proc.TextObject(sentence_list)
################################

######################################################################################
########################### Fit or load embedding model  #############################
######################################################################################

embedding_model_path = os.path.join('saved_models', textname, f"cbow_{embedding_dim}")

if load_embedding_model:
    embedding_model = models.CBOW_model(
        model_path=embedding_model_path,
        text_object=training_text
        )
else:
    embedding_model = models.CBOW_model(
        embedding_dim=embedding_dim,
        text_object=training_text
        )
    embedding_model.fit(training_text.embedding_X, training_text.embedding_Y, epochs=80, verbose=1)
    embedding_model.save(embedding_model_path)


############################################################################################
########################### Fit or load word prediction model  #############################
############################################################################################

PN_model_name = f'PN_{k}_{PN_architecture}'
PN_model_path = os.path.join('saved_models', textname, PN_model_name)
PN_obj = proc.PredictNext(
    embedding_model=embedding_model, text_object=training_text, k=k)
PN_X, PN_Y = PN_obj.returnTrainData()

if load_PN_model:
    PN_model = models.PredictionModel(
        k=k,
        PN_obj=PN_obj,
        model_path=PN_model_path
        )
else:
    PN_model = models.PredictionModel(k=k,
                               PN_obj=PN_obj,
                               architecture=PN_architecture)
    PN_model.fit(PN_X, PN_Y, epochs=40, verbose=1)
    PN_model.save(PN_model_path)

#################################################################################
########################### Generate new sentences  #############################
#################################################################################

with open(os.path.join('outputs', textname, f'{PN_model_name}.txt'), 'w') as f:
    f.writelines([PN_model.genSentence() + '\n' for i in range(50)])
