# Propaganda Detection and Categorization

This repository contains code and data for detecting and categorizing propaganda in text using various NLP techniques. We implement and evaluate the following approaches:
1. TF-IDF with SVM
2. Word2Vec with SVM
3. BERT Sequence Classification

## Data

The `data/` directory contains the dataset used for training and testing. The raw data is stored in `raw/`, and the processed data ready for model training is in `processed/`.

## Notebooks

The `notebooks/` directory contains Jupyter notebooks used for model training and evaluation.

## Scripts

The `scripts/` directory contains standalone Python scripts for preprocessing, training, and evaluating the models.

## Results

Results are saved in the results/ directory in CSV format, showing precision, recall, and F1-score for each approach.

`pip3 install -r requirements.txt`

Gensim library has compatiblity issues with the scipy version required. A [Fix](https://github.com/piskvorky/gensim/issues/3525#issuecomment-2041530964)