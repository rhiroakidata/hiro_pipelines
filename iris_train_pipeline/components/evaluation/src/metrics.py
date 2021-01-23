import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve
from tensorflow.python.lib.io import file_io


def evaluation(prediction, output, labels, y_train, y_scores):
    prediction = np.load(prediction)
    output = np.load(output)
    y_scores = np.load(y_scores)

    # Confusion Matrix
    cm = confusion_matrix(output, prediction, labels)

    data=[]
    for target_index, target_row in enumerate(cm):
        for predicted_index, count in enumerate(target_row):
            data.append((labels[target_index], labels[predicted_index], count))

    df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
    cm_file = os.path.join('evaluation', 'confusion_matrix.csv')
    with file_io.FileIO(cm_file, 'w') as f:
        df_cm.to_csv(f, columns=['target', 'predicted', 'count'], header=False, index=False)

    cm_metadata = {
        'outputs': [{
            'type': 'confusion_matrix',
            'format': 'csv',
            'schema': [
                {'name': 'target', 'type': 'CATEGORY'},
                {'name': 'predicted', 'type': 'CATEGORY'},
                {'name': 'count', 'type': 'NUMBER'},
            ],
            'source': cm_file,
            'labels': labels,
        }]
    }
    with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(cm_metadata, f)

    # Accuracy
    accuracy = accuracy_score(output, prediction)
    acc_metadata = {
        'evaluation': [{
            'name': 'accuracy-score',
            'numberValue': accuracy,
            'format': "PERCENTAGE",
        }]
    }
    with file_io.FileIO('/mlpipeline-evaluation.json', 'w') as f:
        json.dump(acc_metadata, f)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_train, y_scores)

    df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
    roc_file = os.path.join('evaluation', 'roc.csv')
    with file_io.FileIO(roc_file, 'w') as f:
        df_roc.to_csv(f, columns=['fpr', 'tpr', 'thresholds'], header=False, index=False)

    roc_metadata = {
        'outputs': [{
            'type': 'roc',
            'storage': 'gcs',
            'format': 'csv',
            'schema': [
                {'name': 'fpr', 'type': 'NUMBER'},
                {'name': 'tpr', 'type': 'NUMBER'},
                {'name': 'thresholds', 'type': 'NUMBER'},
            ],
            'source': roc_file
        }]
    }

    with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(roc_metadata, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions')
    parser.add_argument('--output')
    parser.add_argument('--labels')
    parser.add_argument('--y_train')
    parser.add_argument('--y_scores')
    args = parser.parse_args()
    evaluation(args.predictions, args.output, args.labels)

