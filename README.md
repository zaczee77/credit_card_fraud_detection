# Credit card fraud detection

A small machine learning project by Nirmeet Rao that trains a logistic regression classifier to flag fraudulent credit card transactions. Three short scripts carry the whole pipeline: prepare the data, train and evaluate the model, and plot the transaction amounts. Everything runs in plain Python with pandas, scikit learn, matplotlib, and seaborn.

## What is in this repository

1. preprocess_data.py. Loads the transaction table, standardizes the Amount and Time columns with a standard scaler, saves the result as a preprocessed file, then prints a summary of the preprocessed table.
2. model.py. Loads the preprocessed table, treats every column except Class as a feature and Class as the label, splits the rows 70 percent train and 30 percent test with a fixed seed of 42, fits a logistic regression model, and prints a confusion matrix and a classification report.
3. visualize.py. Loads the preprocessed table and draws a histogram of transaction amounts with a smooth density curve, so the shape of the spending distribution is easy to see at a glance.

## How it works

The pipeline runs in three steps. First the raw transaction file is cleaned and scaled, because Amount and Time live on very different scales and the model learns better when they are comparable. Then the model is trained on most of the rows and tested on the rest, which shows how it behaves on transactions it has not seen. Finally the amount distribution is plotted, which is a quick sanity check that the data looks the way you expect before you trust any model built on it.

## How to run it

1. Place the transaction data at data/creditcard.csv. The scripts expect a table with a Class label column and Amount and Time columns. The data file itself is not in this repository, so this step comes first.
2. Install the packages.
3. Run the scripts in order.

```
pip install pandas scikit-learn matplotlib seaborn
```

```
python preprocess_data.py
python model.py
python visualize.py
```

## What the model reports

The model prints a confusion matrix and a classification report on the held out test rows. No accuracy, precision, recall, or F1 numbers are claimed here, because none were recorded when the scripts were written. With rare events like fraud, a single accuracy number can hide misses, so read the confusion matrix first: it shows how many fraudulent transactions were caught and how many were missed.

## What this is not

This repository is not production fraud software. It makes no claim about detection performance, it connects to no live data, and it ships no deployment. It is a compact, readable example of the standard fraud detection workflow: scale the features, fit a simple classifier, and look at the errors before anything else.

## Links

1. Portfolio: https://nirmeetrao.com
2. GitHub: https://github.com/Zaczee77
3. LinkedIn: https://www.linkedin.com/in/nirmeet-rao
