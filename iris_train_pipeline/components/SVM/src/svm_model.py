from sklearn import svm
from sklearn.model_selection import cross_val_predict, StratifiedKFold

import joblib, argparse
import numpy as np

def fit_svm(svm_filename, X_train, y_train, X_test, kernel, C, n_splits):
    svm_model = svm.SVC(kernel = kernel, C=C)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    y_scores = cross_val_predict(svm_model, X_train, y_train, cv = cv,
                             method = 'decision_function')

    svm_fit = svm_model.fit(X_train, y_train)

    svm_predict = svm_fit.predict(X_test)

    svm_converted = joblib.dump(svm_fit, svm_filename)

    np.save("svm_model.npy", svm_converted)
    np.save("svm_predict.npy", svm_predict)
    np.save("svm_y_scores.npy", y_scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--svm_filename')
    parser.add_argument('--X_train')
    parser.add_argument('--y_train')
    parser.add_argument('--X_test')
    parser.add_argument('--kernel')
    parser.add_argument('--C')
    args = parser.parse_args()

    print('Training SVM...')
    fit_svm(args.X_train, args.y_train, args.X_test, args.kernel, args.C)

