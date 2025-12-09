import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import paths
import pandas as pd
if __name__ == '__main__':

    # Load preprocessed data
    train_data = pd.read_csv(paths.TRAIN_DATA_PREPROCESSED)
    test_data = pd.read_csv(paths.TEST_DATA_PREPROCESSED)

    # Extract features and labels
    X_train = train_data['review']
    y_train = train_data['sentiment']
    X_test = test_data['review']
    y_test = test_data['sentiment']
    # Load the best Logistic Regression model
    with open(paths.BLM_MODEL_PATH, 'rb') as file:
        best_model = pickle.load(file)

    # Predict on the test set
    y_pred = best_model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()