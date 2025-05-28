# Text Generation with CBOW and Predict-Next Models

This project implements a two-stage NLP pipeline that learns word embeddings using a Continuous Bag of Words (CBOW) model, and then trains a next-word prediction model to generate natural language sentences. Built with TensorFlow, the project demonstrates key NLP and neural language modeling techniques.

## Project Structure

- `data/`: Raw input text files (e.g., mockingbird.txt)
- `outputs/`: Generated sentence outputs
- `saved_models/`: Trained model files
- `src/models.py`: CBOW and prediction model classes
- `src/process_data.py`: Text preprocessing and training data generation
- `main.py`: Main script to run the full pipeline
- `README.md`: This file

## Features

- Text preprocessing and tokenization
- CBOW model for learning word embeddings
- Prediction model using sliding windows of word embeddings
- Sentence generation using the trained models

## Getting Started

1. Install dependencies:

    ```bash
    pip install numpy pandas tensorflow
    ```

2. Add your training text to the `data/` folder (e.g., `mockingbird.txt`).

3. Run the pipeline:

    ```bash
    python main.py
    ```

4. View the generated sentences in the `outputs/` directory.

## Configurable Parameters (in `main.py`)

- `filename`: Input text file name
- `embedding_dim`: Size of word embeddings
- `k`: Number of context words used for prediction
- `PN_architecture`: Hidden layer sizes for the prediction model
- `load_embedding_model`: Whether to load or train the CBOW model
- `load_PN_model`: Whether to load or train the next-word prediction model

## License

This project is open-source and available under the MIT License.
