import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import paths

if __name__ == '__main__':
    # Load balanced training data
    with open(paths.DATA_X_TRAIN_BALANCED_PICKLE_PATH, 'rb') as file:
        X_train = pickle.load(file)
    with open(paths.DATA_Y_TRAIN_BALANCED_PICKLE_PATH, 'rb') as file:
        y_train = pickle.load(file)

    # Load test data
    with open(paths.DATA_PROCESSED_X_TEST, 'rb') as file:
        X_test = pickle.load(file)
    test_data = pd.read_csv(paths.TEST_DATA_PREPROCESSED)
    y_test = test_data['sentiment']

    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save the trained model
    with open(paths.LM_MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(classification_report(y_test, y_pred))