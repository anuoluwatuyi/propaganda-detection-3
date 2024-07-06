# Propaganda Detection and Categorization

This repository contains code and data for detecting and categorizing propaganda in text using various NLP techniques. We implement and evaluate the following approaches:
1. TF-IDF with SVM
2. Word2Vec with SVM
3. BERT Sequence Classification

This research focuses on the detection of propaganda within text, a crucial task for ensuring information integrity across various media platforms. Our study establishes a baseline using Support Vector Machines (SVM) combined with TF-IDF vectors, followed by more sophisticated models incorporating Word2Vec embeddings and BERT sequence classification. The SVM model with Word2Vec targets contextual word similarities while the BERT sequence classification model aims for comprehensive semantic analysis. We assessed the performance of each model based on precision, recall, and F1-score metrics. The findings demonstrate that the BERT sequence classification model surpasses the TF-IDF-SVM and Word2Vec-SVM combinations. Although the latter two models achieved comparable F1-scores of 0.68 and 0.69 respectively on the detection task, they were less effective in classifying propaganda techniques. BERT excelled in both tasks, achieving F1-scores of 0.94 in propaganda detection and 0.78 in propaganda categorization, offering enhanced detection capabilities for complex propaganda techniques. This suggests that BERT-based models are more effective in the nuanced identification of propaganda, promoting more reliable and precise text classification.

_Find full report __[here](https://github.com/anuoluwatuyi/propaganda-detection-3/blob/main/report/Propaganda_Detection.pdf)___


<!-- ## Data

The `data/` directory contains the dataset used for training and testing. The raw data is stored in `raw/`, and the processed data ready for model training is in `processed/`.

## Notebooks

The `notebooks/` directory contains Jupyter notebooks used for model training and evaluation.

## Scripts

The `scripts/` directory contains standalone Python scripts for preprocessing, training, and evaluating the models.

## Results

Results are saved in the results/ directory in CSV format, showing precision, recall, and F1-score for each approach.

`pip3 install -r requirements.txt`

Gensim library has compatiblity issues with the scipy version required. A [Fix](https://github.com/piskvorky/gensim/issues/3525#issuecomment-2041530964) -->
