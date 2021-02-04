import click, os
from click.testing import CliRunner
import numpy as np

from src.logistic_regression_model import fit_lr

def test_fit_lr():

    runner = CliRunner()

    x_train_path = "../../transform/src/X_train.npy"
    y_train_path = "../../transform/src/y_train.npy"
    x_test_path = "../../transform/src/X_test.npy"

    with open(x_train_path, "rb") as f:
        X_train = np.load(f, allow_pickle=True)

    with open(y_train_path, "rb") as f:
        y_train = np.load(f, allow_pickle=True)

    with open(x_test_path, "rb") as f:
        X_test = np.load(f, allow_pickle=True)

    with runner.isolated_filesystem():

        np.save("X_train.npy", X_train)
        np.save("y_train.npy", y_train)
        np.save("X_test.npy", X_test)
            
        result = runner.invoke(
            fit_lr, [
                '--X_train', "X_train.npy",
                '--y_train', "y_train.npy",
                '--X_test', "X_test.npy",
                '--lr_filename', 'lr_model.pkl',
                '--n_splits', 3
            ] 
        )

        assert result.exit_code==0

        assert set([
            'lr_model.npy',
            'lr_predict.npy',
            'lr_y_scores.npy']
            ).intersection(os.listdir())

