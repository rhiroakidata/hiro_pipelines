import click, os
from click.testing import CliRunner
import numpy as np

from src.metrics import evaluation

def test_evaluation():

    runner = CliRunner()

    with open("../src/svm_predict.npy", "rb") as f:
        svm_predict = np.load(f, allow_pickle=True)

    with open("../../transform/src/y_train.npy", "rb") as f:
        y_train = np.load(f, allow_pickle=True)

    with open("../../transform/src/y_test.npy", "rb") as f:
        y_test = np.load(f, allow_pickle=True)

    with open("../src/svm_y_scores.npy", "rb") as f:
        svm_y_scores = np.load(f, allow_pickle=True)

    with runner.isolated_filesystem():

        np.save("svm_predict.npy", svm_predict)
        np.save("y_train.npy", y_train)
        np.save("y_test.npy", y_test)
        np.save("svm_y_scores.npy", svm_y_scores)

        print("-1")
        os.mkdir('metrics')
        print(os.listdir())
            
        result = runner.invoke(
            evaluation, [
                '--prediction', 'svm_predict.npy',
                '--output', 'y_test.npy',
                '--labels', ['setosa', 'versicolor', 'virginica'],
                '--y_train', 'y_train.npy',
                '--y_scores', 'svm_y_scores.npy'
            ] 
        )

        print(result.output)
        print(os.listdir())
        print("**************************************")        

        assert result.exit_code==0


        assert set([
            'mlpipeline-ui-metadata.json',
            'mlpipeline-evaluation.json'
            ]).intersection(os.listdir("/tmp/"))

        assert set([
            'confusion_matrix.csv'
            ]).intersection(os.listdir("metrics/"))

