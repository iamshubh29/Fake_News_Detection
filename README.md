# Fake News Detection

This project aims to classify news articles as either **real** or **fake** using machine learning techniques. The dataset includes news article information such as title, author, and content, and the model is trained to predict whether an article is fake or real.

## Dataset Description

The dataset consists of the following columns:

- `id`: Unique identifier for each news article
- `title`: The title of the news article
- `author`: The author of the news article
- `text`: The content of the news article (may be incomplete)
- `label`: 
  - `1`: Fake news
  - `0`: Real news

## Steps for Data Pre-processing

1. **Handling Missing Data:**
   - Any missing values in the dataset are replaced with an empty string.
   
2. **Merging Columns:**
   - The `author` and `title` columns are merged into a new column `content`.

3. **Stemming:**
   - Stemming is applied to the content to reduce words to their root form. For example: "acting", "actor", "actress" become "act".
   - The `PorterStemmer` from `nltk` library is used for stemming, and common English stopwords are removed.

## Model Building

- **Vectorization:**
  - The `TfidfVectorizer` is used to convert the textual data into numerical format for the machine learning model.

- **Splitting the Dataset:**
  - The dataset is split into training and test sets with an 80:20 ratio, using `train_test_split`.

- **Logistic Regression Model:**
  - A logistic regression model is trained on the dataset.

## Evaluation

- **Accuracy on Training Data:**
  - After training, the model is evaluated on the training data for accuracy.

- **Accuracy on Test Data:**
  - The model is also evaluated on the test data to check generalization performance.

## Libraries Used

- `nltk` for natural language processing
- `numpy` and `pandas` for data handling
- `scikit-learn` for machine learning and model evaluation


