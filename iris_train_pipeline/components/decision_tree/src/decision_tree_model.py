import click, joblib
import numpy as np

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

@click.command()
@click.option('--X_train')
@click.option('--y_train')
@click.option('--X_test')
@click.option('--y_train')
@click.option('--dt_filename')
@click.option('--n_splits')
def fit_decision_tree(x_train, y_train, x_test, dt_filename, n_splits):
    dt_model = DecisionTreeClassifier()

    x_train = np.load(x_train, allow_pickle=True)
    y_train = np.load(y_train, allow_pickle=True)
    x_test = np.load(x_test, allow_pickle=True)

    cv = StratifiedKFold(n_splits=int(n_splits), shuffle=True)
    y_scores = cross_val_predict(dt_model, x_train, y_train, cv = cv)

    dt_fit = dt_model.fit(x_train, y_train)

    dt_predict = dt_fit.predict(x_test)

    dt_converted = joblib.dump(dt_fit, dt_filename)

    np.save("dt_model.npy", dt_converted)
    np.save("dt_predict.npy", dt_predict)
    np.save("dt_y_scores.npy", y_scores)

if __name__ == '__main__':
    print('Training Decision Tree...')
    fit_decision_tree()

