# HW-4: Modeling Natural Language Data

## Overview
This project focuses on natural language processing (NLP) using the 20 Newsgroups dataset. The dataset consists of approximately 12,000 newsgroup posts across 20 topics. The goal is to process, vectorize, and model this textual data to extract insights using various NLP techniques.

## Objectives
1. **Text Preprocessing**
   - Tokenization
   - Noise Removal (stopwords, punctuation, HTML tags)
   - Normalization (lowercasing, lemmatization)
2. **Bag-of-Words (BoW) Representation**
   - Create BoW vectors using `CountVectorizer`.
   - Compute vocabulary size and display a sample vector.
3. **Bigrams Extension**
   - Extend BoW with bigrams.
   - Compare vocabulary size with unigram BoW.
4. **Topic Modeling with LDA**
   - Implement Latent Dirichlet Allocation (LDA) to discover 10 topics.
   - Use Gensim’s `LdaModel` and visualize the topics.
5. **Word Embeddings with GloVe**
   - Load pre-trained GloVe embeddings.
   - Compute document embeddings by averaging word vectors.
6. **Document Similarity**
   - Compute cosine similarity between document embeddings.
   - Identify the most similar document for a given query.

## Installation & Setup
1. Install required Python packages:
   ```bash
   pip install numpy pandas nltk scikit-learn gensim matplotlib pyLDAvis
   ```
2. Download the 20 Newsgroups dataset:
   ```python
   from sklearn.datasets import fetch_20newsgroups
   newsgroups_data = fetch_20newsgroups(subset='train')
   documents = newsgroups_data.data
   ```
3. Download and extract [GloVe embeddings](https://nlp.stanford.edu/data/glove.6B.zip), using `glove.6B.50d.txt` for this project.

## Usage
- Run the provided Jupyter Notebook step by step to execute the entire NLP pipeline.
- Ensure that `glove.6B.50d.txt` is placed in the correct directory before running the embedding section.
- Modify the preprocessing pipeline as needed to experiment with different techniques.

## Results
- Tokenized and cleaned text data.
- BoW and TF-IDF feature representations.
- Extended BoW model with bigrams for improved textual representation.
- Topics extracted from LDA with visualizations.
- Document embeddings and similarity analysis.

## Acknowledgments
- [Scikit-learn](https://scikit-learn.org/) for dataset and vectorization tools.
- [Gensim](https://radimrehurek.com/gensim/) for LDA modeling.
- [NLTK](https://www.nltk.org/) for text preprocessing.
- [Stanford NLP](https://nlp.stanford.edu/projects/glove/) for GloVe embeddings.

## Author
Neela Ropp

## License
This project is for educational purposes and follows an open-source approach. Feel free to modify and experiment with the code!

