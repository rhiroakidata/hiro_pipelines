import click, os
from click.testing import CliRunner
import numpy as np

from src.svm_model import fit_svm

def test_fit_svm():

    runner = CliRunner()

    with open("../../transform/src/X_train.npy", "rb") as f:
        X_train = np.load(f, allow_pickle=True)

    with open("../../transform/src/y_train.npy", "rb") as f:
        y_train = np.load(f, allow_pickle=True)

    with open("../../transform/src/X_test.npy", "rb") as f:
        X_test = np.load(f, allow_pickle=True)

    with runner.isolated_filesystem():

        np.save("X_train.npy", X_train)
        np.save("y_train.npy", y_train)
        np.save("X_test.npy", X_test)
            
        result = runner.invoke(
            fit_svm, [
                '--svm_filename', 'svm_model.pkl',
                '--X_train', 'X_train.npy',
                '--y_train', 'y_train.npy',
                '--X_test', 'X_test.npy',
                '--kernel', 'linear',
                '--C', 1,
                '--n_splits', 3
            ] 
        )

        assert result.exit_code==0

        assert set([
            'svm_model.npy',
            'svm_predict.npy',
            'svm_y_scores.npy']
            ).intersection(os.listdir())

