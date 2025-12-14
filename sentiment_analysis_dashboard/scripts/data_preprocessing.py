import pandas as pd
from sklearn.model_selection import train_test_split
import paths
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import pickle
from imblearn.over_sampling import SMOTE

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
_LEMMATIZER = WordNetLemmatizer()
_ENGLISH_STOPWORDS=stopwords.words('english')

# Define preprocessing function
def preprocess_text(text):
    # Initialize lemmatizer
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Remove punctuation and stop words
    tokens = [word for word in tokens if word not in string.punctuation and word not in _ENGLISH_STOPWORDS]
    # Lemmatize tokens
    tokens = [_LEMMATIZER.lemmatize(word) for word in tokens]
    print("Tokens:", tokens)
    return ' '.join(tokens)

def balance_data(vectorizer):
    # Load preprocessed data
    train_data = pd.read_csv(paths.TRAIN_DATA_PREPROCESSED)
    test_data = pd.read_csv(paths.TEST_DATA_PREPROCESSED)

    # Extract features and labels
    x_train = train_data['review']
    y_train = train_data['sentiment']
    x_test = test_data['review']
    y_test = test_data['sentiment']

    # Vectorize the preprocessed text
    x_train_vectorized = vectorizer.fit_transform(x_train).toarray()
    x_test_vectorized = vectorizer.transform(x_test).toarray()

    # Balance the dataset using SMOTE
    smote = SMOTE(random_state=42)
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train_vectorized, y_train)

    # Save the balanced data
    with open(paths.DATA_X_TRAIN_BALANCED_PICKLE_PATH, 'wb') as file:
        pickle.dump(x_train_balanced, file)
    with open(paths.DATA_Y_TRAIN_BALANCED_PICKLE_PATH, 'wb') as file:
        pickle.dump(y_train_balanced, file)


def dowload_and_load_dataset():
    # Load the IMDB Movie Reviews dataset
    data = pd.read_csv(paths.RAW_DATA_PATH)

    # Display the first few rows of the dataset
    print(data.head())

    # Split the dataset into training and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Save the training and test sets
    train_data.to_csv(paths.TRAIN_DATA_PATH, index=False)
    test_data.to_csv(paths.TEST_DATA_PATH, index=False)

    return train_data, test_data

def preprocess_data(train_data, test_data):
    # Apply preprocessing to training and test sets
    train_data['review'] = train_data['review'].apply(preprocess_text)
    test_data['review'] = test_data['review'].apply(preprocess_text)

    # Save preprocessed data
    train_data.to_csv(paths.TRAIN_DATA_PREPROCESSED, index=False)
    test_data.to_csv(paths.TEST_DATA_PREPROCESSED, index=False)

    # Vectorize the preprocessed text
    vectorizer = TfidfVectorizer(max_features=5000)
    x_train = vectorizer.fit_transform(train_data['review']).toarray()
    x_test = vectorizer.transform(test_data['review']).toarray()

    # Save the vectorizer and vectorized data
    with open(paths.MODEL_VECTORIZER, 'wb') as file:
        pickle.dump(vectorizer, file)
    with open(paths.DATA_PROCESSED_X_TRAIN, 'wb') as file:
        pickle.dump(x_train, file)
    with open(paths.DATA_PROCESSED_X_TEST, 'wb') as file:
        pickle.dump(x_test, file)

    return vectorizer


if __name__ == '__main__':
    train_data, test_data = dowload_and_load_dataset()
    #vectorizer = preprocess_data(train_data, test_data)
    vectorizer = pickle.load(open(paths.MODEL_VECTORIZER, 'rb'))
    balance_data(vectorizer)