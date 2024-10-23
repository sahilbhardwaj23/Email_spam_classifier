# Email Spam Classifier

## Project Overview

This project focuses on building an **Email Spam Classifier** that distinguishes between spam and non-spam emails with high precision and accuracy. The classifier uses multiple datasets merged from **Kaggle**, along with various text preprocessing and feature engineering techniques to standardize the data before training machine learning models.

## Key Features

- **Data Preprocessing**: Merging datasets, cleaning text, and removing irrelevant information such as URLs, email addresses, and HTML tags.
- **Feature Engineering**: Extracting additional features like word count, character count, and stopwords count before and after preprocessing.
- **Contextual Embedding**: Using **Word2Vec** to create contextual embeddings of text for more accurate model input.
- **Handling Imbalanced Data**: Techniques like undersampling, oversampling, and SMOTE to balance the class distribution.
- **Model Training**: Training models such as Logistic Regression, Random Forest, XGBoost, and Decision Tree classifiers. The **Random Forest Classifier** showed the best performance.
- **Hyperparameter Tuning**: Fine-tuning the Random Forest model for optimal performance.

## Preprocessing and Feature Engineering

### Preprocessing

A function `preprocess_and_embed` performs the following steps:

1. **Lowercase Conversion**: Converts text to lowercase for uniformity.
2. **Whitespace & Newline Removal**: Cleans text by removing unnecessary whitespaces and newlines.
3. **Stopwords Removal**: Removes common stopwords, while retaining essential ones like 'not' and 'but.'
4. **Lemmatization**: Reduces words to their base forms to avoid redundancy.
5. **Text Cleaning**: Removes URLs, email addresses, HTML tags, and non-English characters.

### Feature Engineering

In addition to text preprocessing, the following features were engineered:

- **Total Word Count**: Calculated both before and after preprocessing.
- **Character Count**: Measured the total number of characters before and after cleaning.
- **Stopwords Count**: Counted the number of stopwords present in the text, both before and after preprocessing.
- **Punctuation Count**: Extracted punctuation character count as a feature.

### Word2Vec Embedding

The `Word2Vec` model was used to convert the tokenized text into contextual embeddings, creating a 100-dimensional vector for each email. This embedded vector helps capture the semantic meaning of words in the text, improving the model's ability to classify emails accurately.


## Challenges Faced

### Imbalanced Dataset
Handling imbalanced datasets is a common problem in spam classification. To address this, the following techniques were used:
- **Undersampling**: Reducing the majority class to balance the data.
- **Oversampling**: Increasing instances of the minority class (spam).
- **SMOTE**: Creating synthetic samples of the minority class.

### Model Performance
Various models were trained and evaluated:
- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost Classifier**
- **Decision Tree Classifier**

After testing, the **Random Forest Classifier** yielded the best results in terms of precision, accuracy, and minimizing both Type-1 and Type-2 errors.

## Model Performance & Hyperparameter Tuning

The Random Forest classifier was tuned with the following hyperparameters:

- **n_estimators**: 100
- **max_depth**: 20
- **min_samples_split**: 5
- **min_samples_leaf**: 2
- **max_features**: 'sqrt'
- **class_weight**: 'balanced'

This model performed exceptionally well with precision and accuracy above **0.90**.

## Results

- **Random Forest Classifier**: The best-performing model with high precision, accuracy, and F1-score after hyperparameter tuning.
- **Logistic Regression**: Also performed well after handling class imbalance, with precision and accuracy around **0.90**.

## Future Work

- Explore more advanced models such as **LSTM** or **Transformer-based models** for email classification.
- Improve contextual embeddings using pre-trained models like **BERT**.
- Experiment with other data augmentation techniques for handling class imbalance.

## Conclusion

This project demonstrates an effective approach to email spam classification, incorporating thorough preprocessing, feature engineering, and handling of imbalanced data. The **Random Forest Classifier**, after hyperparameter tuning, emerged as the best-performing model, providing high accuracy in spam detection.
