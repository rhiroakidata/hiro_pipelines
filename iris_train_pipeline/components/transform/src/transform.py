import argparse, click
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

@click.command()
@click.option('--iris')
def transform_data(iris):
    iris = np.load(iris, allow_pickle=True)

    iris = pd.DataFrame(iris, columns=[
        'sepal length (cm)', 
        'sepal width (cm)', 
        'petal length (cm)', 
        'petal width (cm)',
        'species_num',
        'species'
        ])

    data = iris.iloc[:, :-1]
    target = iris.species

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

    # return X_train, y_train, X_test, y_test
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--iris')
    # args = parser.parse_args()

    print('Transforming datas...')
    # transform_data(args.iris)
    transform_data()