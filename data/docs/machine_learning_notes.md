# Machine Learning Notes

## Overfitting

Overfitting happens when a machine learning model learns noise or accidental patterns in the training data instead of learning patterns that generalize to new data. A model is often overfitting when training performance is strong but validation or test performance is much worse.

Common signs include a large gap between training and validation metrics, unstable predictions on small input changes, and performance that degrades when the model sees fresh production data.

## How to Avoid Overfitting

Useful techniques include cross-validation, early stopping, simpler model architectures, more representative training data, feature selection, and regularization. L1 regularization can push unnecessary feature weights toward zero, while L2 regularization discourages very large weights.

Data leakage should also be checked because it can make validation scores look unrealistically good. Keep preprocessing and feature engineering inside the training pipeline so validation folds do not see information from the future.

## ROC Curve

The ROC curve plots true positive rate against false positive rate across classification thresholds. The area under the curve, or AUC, is often used to summarize how well a classifier separates positive and negative classes.

## Cross-Validation

Cross-validation estimates model performance by training and evaluating on multiple train-validation splits. It is useful when data is limited and helps detect whether results are stable across different subsets of data.

