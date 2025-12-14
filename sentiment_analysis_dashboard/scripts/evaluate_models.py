import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import paths

# Load test data and preprocessed text
test_data = pd.read_csv(paths.TEST_DATA_PREPROCESSED)
with open(paths.DATA_PROCESSED_X_TEST, 'rb') as file:
    X_test = pickle.load(file)

y_test = test_data['sentiment']

# Load Logistic Regression model
with open(paths.BLM_MODEL_PATH, 'rb') as file:
    logistic_regression_model = pickle.load(file)

# Evaluate Logistic Regression model
y_pred_lr = logistic_regression_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f'Logistic Regression Accuracy: {accuracy_lr}')
print(classification_report(y_test, y_pred_lr))
cm_lr = confusion_matrix(y_test, y_pred_lr)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['Negative', 'Positive'])
disp_lr.plot(cmap=plt.cm.Blues)
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# Load LSTM model
from tensorflow.keras.models import load_model
lstm_model = load_model(paths.LSTM_MODEL_PATH)

# Evaluate LSTM model
y_pred_prob_lstm = lstm_model.predict(X_test)
y_pred_lstm = (y_pred_prob_lstm > 0.5).astype(int)
ytest_lstm = y_test.apply(lambda x: 1 if x == 'positive' else 0)
accuracy_lstm = accuracy_score(ytest_lstm, y_pred_lstm)
print(f'LSTM Accuracy: {accuracy_lstm}')
print(classification_report(ytest_lstm, y_pred_lstm))
cm_lstm = confusion_matrix(ytest_lstm, y_pred_lstm, labels=[0, 1])
disp_lstm = ConfusionMatrixDisplay(confusion_matrix=cm_lstm, display_labels=['Negative', 'Positive'])
disp_lstm.plot(cmap=plt.cm.Blues)
plt.title('LSTM Confusion Matrix')
plt.show()