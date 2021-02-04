from sklearn import svm
from sklearn.model_selection import cross_val_predict, StratifiedKFold

import joblib, click
import numpy as np

@click.command()
@click.option('--svm_filename')
@click.option('--X_train')
@click.option('--y_train')
@click.option('--X_test')
@click.option('--kernel')
@click.option('--C')
@click.option('--n_splits')
def fit_svm(svm_filename, x_train, y_train, x_test, kernel, c, n_splits):
    svm_model = svm.SVC(kernel = kernel, C=int(c))
    cv = StratifiedKFold(n_splits=int(n_splits), shuffle=True)

    x_train = np.load(x_train, allow_pickle=True)
    y_train = np.load(y_train, allow_pickle=True)
    x_test = np.load(x_test, allow_pickle=True)

    y_scores = cross_val_predict(
        estimator=svm_model,
        X=x_train,
        y=y_train,
        cv = cv,
        method = 'decision_function'
    )

    svm_fit = svm_model.fit(
        X=x_train,
        y=y_train
    )

    svm_predict = svm_fit.predict(x_test)

    svm_converted = joblib.dump(svm_fit, svm_filename)

    np.save("svm_model.npy", svm_converted)
    np.save("svm_predict.npy", svm_predict)
    np.save("svm_y_scores.npy", y_scores)

if __name__ == '__main__':
    print('Training SVM...')
    fit_svm()

