import argparse, joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold

def fit_lr(X_train, y_train, X_test, lr_filename, n_splits):
    lr_model = LogisticRegression()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    y_scores = cross_val_predict(lr_model, X_train, y_train, cv=cv,
                                 method='decision_function')

    lr_fit = lr_model.fit(X_train, y_train)

    lr_predict = lr_fit.predict(X_test)

    lr_converted = joblib.dump(lr_fit, lr_filename)

    np.save("lr_model.npy", lr_converted)
    np.save("lr_predict.npy", lr_predict)
    np.save("lr_y_scores.npy", y_scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--X_train')
    parser.add_argument('--y_train')
    parser.add_argument('--X_test')
    parser.add_argument('--lr_filename')
    args = parser.parse_args()

    print('Training Logistic Regression...')
    fit_lr(args.X_train, args.y_train, args.X_test)

