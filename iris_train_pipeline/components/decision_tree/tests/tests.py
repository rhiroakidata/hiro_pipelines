import click, os
from click.testing import CliRunner
import numpy as np

from src.decision_tree_model import fit_decision_tree

def test_fit_decision_tree():

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
            fit_decision_tree, [
                '--X_train', 'X_train.npy',
                '--y_train', 'y_train.npy',
                '--X_test', 'X_test.npy',
                '--dt_filename', 'dt_model.pkl',
                '--n_splits', 3
            ] 
        )

        print(result.exception)
        print(result.stdout)

        assert result.exit_code==0

        assert set([
            'dt_model.npy',
            'dt_predict.npy',
            'dt_y_scores.npy']
            ).intersection(os.listdir())

