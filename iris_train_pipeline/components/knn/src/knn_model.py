from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold

import click, joblib
import numpy as np

@click.command()
@click.option('--X_train')
@click.option('--y_train')
@click.option('--X_test')
@click.option('--n_neighbors')
@click.option('--n_splits')
@click.option('--knn_filename')
def fit_knn(x_train, y_train, x_test, n_neighbors, n_splits, knn_filename):
    knn_model = KNeighborsClassifier(n_neighbors=int(n_neighbors))
    cv = StratifiedKFold(n_splits=int(n_splits), shuffle=True)
    x_train = np.load(x_train, allow_pickle=True)
    y_train = np.load(y_train, allow_pickle=True)
    x_test = np.load(x_test, allow_pickle=True)

    y_scores = cross_val_predict(knn_model, x_train, y_train, cv=cv)

    knn_fit = knn_model.fit(x_train, y_train)

    knn_predict = knn_fit.predict(x_test)

    knn_converted = joblib.dump(knn_fit, knn_filename)

    np.save("knn_model.npy", knn_converted)
    np.save("knn_predict.npy", knn_predict)
    np.save("knn_y_scores.npy", y_scores)

if __name__ == '__main__':
    print('Training KNN...')
    fit_knn()

