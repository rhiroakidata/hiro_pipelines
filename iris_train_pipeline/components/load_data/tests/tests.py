import numpy as np
import os
from src.load_data import load_data

def test_load_data():
    iris = load_data()

    assert iris.shape==(150,6)

    assert sorted(list(set(iris.species))) == ['setosa', 'versicolor', 'virginica']

    assert 'iris.npy' in os.listdir()
    os.remove('iris.npy')