from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold

import argparse, joblib
import numpy as np

def fit_knn(X_train, y_train, X_test, n_neighbors, n_splits, knn_filename):
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    y_scores = cross_val_predict(knn_model, X_train, y_train, cv=cv,
                                 method='decision_function')

    knn_fit = knn_model.fit(X_train, y_train)

    knn_predict = knn_fit.predict(X_test)

    knn_converted = joblib.dump(knn_fit, knn_filename)

    np.save("knn_model.npy", knn_converted)
    np.save("knn_predict.npy", knn_predict)
    np.save("knn_y_scores.npy", y_scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--X_train')
    parser.add_argument('--y_train')
    parser.add_argument('--X_test')
    parser.add_argument('--n_neighbors')
    parser.add_argument('--knn_filename')
    args = parser.parse_args()

    print('Training KNN...')
    fit_knn(args.X_train, args.y_train, args.X_test, args.kernel, args.n_neighbors)

