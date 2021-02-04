import click, joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold

@click.command()
@click.option('--X_train')
@click.option('--y_train')
@click.option('--X_test')
@click.option('--lr_filename')
@click.option('--n_splits')
def fit_lr(x_train, y_train, x_test, lr_filename, n_splits):
    lr_model = LogisticRegression()

    cv = StratifiedKFold(n_splits=int(n_splits), shuffle=True)
    
    x_train = np.load(x_train, allow_pickle=True)
    y_train = np.load(y_train, allow_pickle=True)
    x_test = np.load(x_test, allow_pickle=True)

    y_scores = cross_val_predict(
        lr_model,
        x_train,
        y_train,
        cv=cv,
        method='decision_function')

    lr_fit = lr_model.fit(x_train, y_train)

    lr_predict = lr_fit.predict(x_test)

    lr_converted = joblib.dump(lr_fit, lr_filename)

    np.save("lr_model.npy", lr_converted)
    np.save("lr_predict.npy", lr_predict)
    np.save("lr_y_scores.npy", y_scores)

if __name__ == '__main__':
    print('Training Logistic Regression...')
    fit_lr()

