import numpy as np
from sklearn.datasets import load_iris
from pandas import DataFrame
import pandas as pd

def load_data():
    iris = load_iris()
    transform = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

    # Convert to Dataframe
    iris = DataFrame(
        iris.data,
        columns=iris.feature_names,
        index=pd.Index([i for i in range(iris.data.shape[0])])).join(DataFrame(
            iris.target,
            columns=pd.Index(["species"]),
            index=pd.Index([i for i in range(iris.target.shape[0])])
            ))
    
    # Rename column species to species_num
    iris.rename(columns={'species':'species_num'}, inplace=True)

    # Add column species through transform dict
    iris['species'] = iris['species_num'].map(transform)

    # Save in format numpy
    np.save('iris.npy', iris)
    return iris

if __name__ == '__main__':
    print('Loading data...')
    load_data()