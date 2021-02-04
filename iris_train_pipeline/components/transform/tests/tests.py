import click, os
from click.testing import CliRunner
import numpy as np

from src.transform import transform_data

def test_transform_data():

    runner = CliRunner()

    with open("transform/src/iris.npy", "rb") as f:
        iris = np.load(f, allow_pickle=True)

    with runner.isolated_filesystem():

        np.save("iris.npy", iris)
            
        result = runner.invoke(
            transform_data, [
                '--iris', "iris.npy"
            ] 
        )

        assert result.exit_code==0

        assert set(['X_train.npy', 'y_train.npy', 'X_test.npy', 'y_test.npy']).intersection(os.listdir())

