import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report
import paths

if __name__ == '__main__':
    # Load preprocessed data
    train_data = pd.read_csv(paths.TRAIN_DATA_PREPROCESSED)
    test_data = pd.read_csv(paths.TEST_DATA_PREPROCESSED)

    # Extract features and labels
    X_train = train_data['review']
    y_train = train_data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    X_test = test_data['review']
    y_test = test_data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    # Tokenize and pad sequences
    tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    word_index = tokenizer.word_index

    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    max_length = 200
    X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding='post', truncating='post')
    X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding='post', truncating='post')

    # Build the LSTM model
    embedding_dim = 100
    model = Sequential([
        Embedding(input_dim=5000, output_dim=embedding_dim, input_length=max_length),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train_padded, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Save the trained model and tokenizer
    model.save(paths.LSTM_MODEL_PATH)
    with open(paths.TOKENIZER_PICKLE_PATH, 'wb') as file:
        pickle.dump(tokenizer, file)

    # Evaluate the model on the test set
    y_pred_prob = model.predict(X_test_padded)
    y_pred = (y_pred_prob > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(classification_report(y_test, y_pred))