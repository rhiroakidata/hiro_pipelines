import argparse
import joblib
import numpy as np

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

def fit_decision_tree(X_train, y_train, X_test, dt_filename, n_splits):
    dt_model = DecisionTreeClassifier()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    y_scores = cross_val_predict(dt_model, X_train, y_train, cv = cv,
                             method = 'decision_function')

    dt_fit = dt_model.fit(X_train, y_train)

    dt_predict = dt_fit.predict(X_test)

    dt_converted = joblib.dump(dt_fit, dt_filename)

    np.save("dt_model.npy", dt_converted)
    np.save("dt_predict.npy", dt_predict)
    np.save("dt_y_scores.npy", y_scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--X_train')
    parser.add_argument('--y_train')
    parser.add_argument('--X_test')
    parser.add_argument('--dt_filename')
    args = parser.parse_args()

    print('Training Decision Tree...')
    fit_decision_tree(args.X_train, args.y_train, args.X_test)

