# Iris Classifier (Decision Tree)

## Overview
This project uses a Decision Tree Classifier built with Scikit-learn to classify iris flowers into three species:
- Setosa
- Versicolor
- Virginica

The model loads the Iris dataset, splits the data into training and testing sets, trains a classifier, evaluates its performance, and generates a confusion matrix.

## Quick Start

```bash
pip install -r requirements.txt
python src/train.py
```

## Project Structure

```text
iris-ai-project/
├── notebooks/
│   └── iris_model.ipynb
├── outputs/
│   └── confusion_matrix.png
├── src/
│   └── train.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Results
The model achieved an accuracy score of 1.0 (100%) on the test dataset.