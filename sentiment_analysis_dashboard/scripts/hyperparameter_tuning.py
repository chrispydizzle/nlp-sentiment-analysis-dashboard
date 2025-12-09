from sklearn.model_selection import GridSearchCV
import paths

if __name__ == '__main__':
    # Define hyperparameters to tune
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear']
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy', verbose=1)

    # Perform hyperparameter tuning
    grid_search.fit(X_train, y_train)

    # Print the best parameters and the corresponding score
    print(f'Best Parameters: {grid_search.best_params_}')
    print(f'Best Score: {grid_search.best_score_}')

    # Save the best model
    best_model = grid_search.best_estimator_
    with open(paths.BLM_MODEL_PATH, 'wb') as file:
        pickle.dump(best_model, file)