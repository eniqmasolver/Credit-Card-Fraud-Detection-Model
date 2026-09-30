![Portada del proyecto](imagen.png)
# Credit Card Fraud Detection Model

This repository contains a credit card fraud detection model implemented in Python using the scikit-learn library. The model is designed to detect fraudulent transactions by employing a Random Forest Classifier, a supervised learning algorithm.

The dataset used for training and testing the model is the Credit Card Fraud Detection dataset sourced from Kaggle. It comprises a total of 284,315 transactions, of which 492 (0.17%) are labeled as fraudulent.

**Dataset Source**: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Model Overview

The model employs a Random Forest Classifier, which excels at handling complex data and is well-suited for classification tasks. To address the imbalanced nature of the credit card fraud dataset, Synthetic Minority Over-sampling Technique (SMOTE) is utilized.

## Model Performance

On the test set, the model demonstrates impressive performance metrics:

- Accuracy: 99.98%
- Precision: 99.97%
- Recall: 1

## Deployment

The model is deployed using Streamlit, a powerful tool for building interactive web applications for machine learning models. Users can input V1-V28 features along with the normalized amount, and the model predicts whether the transaction is fraudulent or legitimate.

> Note: V1-V28 features may be the result of PCA dimensionality reduction to protect user identities and sensitive information.

## Sample Inputs

The Streamlit app expects 29 comma-separated numeric values in this order: `V1` through `V28`, followed by `Normalized_Amount`. Do not include `Time` or feature labels in the input. The final value is the StandardScaler-transformed amount, not the raw `Amount` column.

### Example transaction

```text
-1.359807134, -0.072781173, 2.536346738, 1.378155224, -0.338320770, 0.462387778, 0.239598554, 0.098697901, 0.363786970, 0.090794172, -0.551599533, -0.617800856, -0.991389847, -0.311169354, 1.468176972, -0.470400525, 0.207971242, 0.025790580, 0.403992960, 0.251412098, -0.018306778, 0.277837576, -0.110473910, 0.066928075, 0.128539358, -0.189114844, 0.133558377, -0.021053053, 0.244964263370174
```

### Example fraud transaction

```text
-2.312226542, 1.951992011, -1.609850732, 3.997905588, -0.522187865, -1.426545319, -2.537387306, 1.391657248, -2.770089277, -2.772272145, 3.202033207, -2.899907388, -0.595221881, -4.289253782, 0.389724120, -1.140747180, -2.830055675, -0.016822468, 0.416955705, 0.126910559, 0.517232371, -0.035049369, -0.465211076, 0.320198199, 0.044519167, 0.177839798, 0.261145003, -0.143275874, -0.353229392966824
```

These are sample inputs for demonstration. Model predictions may vary with the model and its training data; they are not a guarantee of a particular result.

## 100-Transaction Dataset Sample

The app bundles [data/sample_transactions.csv](data/sample_transactions.csv), a 100-row sample from the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It contains 90 legitimate and 10 fraud-labeled transactions so both classes are available to explore; this stratified sample is not representative of the full dataset's fraud rate. Expand **Browse 100 sample Kaggle transactions** in the app to filter by label, inspect a row, and load it into the checker.

The file is sourced from [Zenodo's copy of the Kaggle dataset](https://zenodo.org/records/7395559), using a small slice of the original CSV. Selected transaction amounts are normalized with the full-dataset StandardScaler statistics used by `creditcard.ipynb` before prediction. `Class` is shown as the row's known dataset label.

## Conclusion

This fraud detection model showcases the effectiveness of the Random Forest Classifier, especially when combined with techniques like SMOTE for handling imbalanced data. The high accuracy, precision, and recall scores demonstrate its robustness in identifying fraudulent transactions.

---
v2.0
