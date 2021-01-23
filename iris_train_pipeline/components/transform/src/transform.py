import argparse
import numpy as np

from sklearn.model_selection import train_test_split

def transform_data(iris):
    transform = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    iris['species_num'] = iris['species'].map(transform)

    data = iris.iloc[:, :-1]
    target = iris['species']

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iris')
    args = parser.parse_args()

    print('Transforming datas...')
    transform_data(args.iris)