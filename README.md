# Regression on the tabular data
------------------------------------

## Overview

This project focuses on feature analysis, aiming to build a predictive model based on a dataset containing 53 anonymized features and a target column. The primary goal is to analyze and understand the relationships between the features and the target variable for accurate predictions.

*Note*: All presented code was performed with Python 3.8.10.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/RostyslavBryiovskyi/Tabular-data.git
cd Tabular-data
```

2. **Install the requirements:**
```bash
pip install -r requirements.txt
```

## Usage
This folder contains:
* *eda.ipynb* - jupyter notebook with exploratory data analysis;
* *train.py* -  python script for model training;
* *predict.py* - python script for model inference on test data;
* *test_predictions.csv* - file with prediction results;
* *requirements.txt* - requirements to running;

In order to run training from your terminal, specify path to the CSV file containing the dataset and path for store trained model:

```bash
python train.py --data_path train.csv --model_save_path trained_model.joblib
```

To evaluate model on test data and save predictions to file, please run the follow:
```bash
python predict.py --data_path hidden_test.csv --model_path trained_model.joblib --predictions_save_path test_predictions.csv
```
