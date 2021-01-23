from kfp import dsl
from kfp import aws


# ========== Operations ============
def load_op():
    return dsl.ContainerOp(
      name='Load Data',
      image='rhiroakidata/iris_components/load_data:latest',
      arguments=[],
      file_outputs={
        'iris': '/app/iris.npy',
      }
    )

def transform_op(iris):
    return dsl.ContainerOp(
      name='Transform Data',
      image='rhiroakidata/iris_components/transform:latest',
      arguments=[
        '--iris', iris
      ],
      file_outputs={
        'X_train': '/app/X_train.npy',
        'y_train': '/app/y_train.npy',
        'X_test': '/app/X_test.npy',
        'y_test': '/app/y_test.npy',
      }
    )

def svm_op(svm_filename, X_train, y_train, X_test, kernel, C, n_splits):
    return dsl.ContainerOp(
        name='SVM',
        image='rhiroakidata/iris_components/svm_sklearn:latest',
        arguments=[
            '--svm_filename', svm_filename,
            '--X_train', X_train,
            '--y_train', y_train,
            '--X_test', X_test,
            '--kernel', kernel,
            '--C', C,
            '--n_splits', n_splits
        ],
        file_outputs={
            'svm_model': '/app/svm_model.npy',
            'svm_predict': '/app/svm_predict.npy'
        }
    )

def lr_op(X_train, y_train, X_test, lr_filename, n_splits):
    return dsl.ContainerOp(
        name='Logistic Regression',
        image='rhiroakidata/iris_components/logistic_regression_sklearn:latest',
        arguments=[
            '--X_train', X_train,
            '--y_train', y_train,
            '--X_test', X_test,
            '--lr_filename', lr_filename,
            '--n_splits', n_splits
        ],
        file_outputs={
            'lr_model': '/app/lr_model.npy',
            'lr_predict': '/app/lr_predict.npy'
        }
    )

def dt_op(X_train, y_train, X_test, dt_filename, n_splits):
    return dsl.ContainerOp(
        name='Decision Tree',
        image='rhiroakidata/iris_components/decision_tree_sklearn:latest',
        arguments=[
            '--X_train', X_train,
            '--y_train', y_train,
            '--X_test', X_test,
            '--dt_filename', dt_filename,
            '--n_splits', n_splits
        ],
        file_outputs={
            'dt_model': '/app/dt_model.npy',
            'dt_predict': '/app/dt_predict.npy'
        }
    )

def knn_op(X_train, y_train, X_test, n_neighbors, n_splits, knn_filename):
    return dsl.ContainerOp(
        name='KNN',
        image='rhiroakidata/iris_components/knn_sklearn:latest',
        arguments=[
            '--X_train', X_train,
            '--y_train', y_train,
            '--X_test', X_test,
            '--n_neighbors', n_neighbors,
            '--n_splits', n_splits,
            '--knn_filename', knn_filename
        ],
        file_outputs={
            'knn_model': '/app/knn_model.npy',
            'knn_predict': '/app/knn_predict.npy'
        }
    )

def save_s3_op(model, location, model_filenames):
    return dsl.ContainerOp(
        name='Save to S3',
        image='rhiroakidata/iris_components/save_s3:latest',
        arguments=[
            '--model', model,
            '--location', location,
            '--model_filenames', model_filenames
        ],
        file_outputs={
            'model': '/app/model.npy'
        }
    )

def evaluation_op(prediction, output, labels, y_train, y_scores):
    return dsl.ContainerOp(
        name='Evaluation',
        image='rhiroakidata/iris_components/evaluation:latest',
        arguments=[
            '--prediction', prediction,
            '--output', output,
            '--labels', labels,
            '--y_train', y_train,
            '--y_scores', y_scores
        ]
    )


# ========== PIPELINE ==============

@dsl.pipeline(
    name='Iris Train Pipeline Example',
    description='Example with the Iris classification'
)
def iris_train_pipeline(
    kernel: dsl.PipelineParam = dsl.PipelineParam(name='kernel', value='linear, poly, rbf, sigmoid or precomputed'),
    C: dsl.PipelineParam = dsl.PipelineParam(name='C', value='Float value, default value is 1'),
    n_neighbors: dsl.PipelineParam = dsl.PipelineParam(name='n_neighbors', value='int value'),
    n_splits: dsl.PipelineParam = dsl.PipelineParam(name='n_splits', value="Number of splits for fold"),
    location: dsl.PipelineParam = dsl.PipelineParam(name='location', value='FOLDER_NAME_TO_MODELS'),
    svm_filename: dsl.PipelineParam = dsl.PipelineParam(name='svm-filename', value='SVM_NAME'),
    lr_filename: dsl.PipelineParam = dsl.PipelineParam(
        name='logistic-regression-filename',
        value='LOGISTIC_REGRESSION_NAME'
    ),
    dt_filename: dsl.PipelineParam = dsl.PipelineParam(
        name='decision-tree-filename',
        value='DECISION_TREE_NAME'
    ),
    knn_filename: dsl.PipelineParam = dsl.PipelineParam(name='knn-filename', value='KNN_NAME'),
    label1: dsl.PipelineParam = dsl.PipelineParam(name='labels', value='Label 1'),
    label2: dsl.PipelineParam = dsl.PipelineParam(name='labels', value='Label 2'),
    label3: dsl.PipelineParam = dsl.PipelineParam(name='labels', value='Label 3')
):
    _load_data = load_op()

    _transform = transform_op(
        dsl.InputArgumentPath(_load_data.outputs['iris'])
    ).after(_load_data)

    _svm = svm_op(
        str(svm_filename) + '.pkl',
        dsl.InputArgumentPath(_transform.outputs['X_train']),
        dsl.InputArgumentPath(_transform.outputs['y_train']),
        dsl.InputArgumentPath(_transform.outputs['X_test']),
        kernel,
        C,
        n_splits
    ).after(_transform)

    _lr = lr_op(
        dsl.InputArgumentPath(_transform.outputs['X_train']),
        dsl.InputArgumentPath(_transform.outputs['y_train']),
        dsl.InputArgumentPath(_transform.outputs['X_test']),
        str(lr_filename) + '.pkl',
        n_splits
    ).after(_transform)

    _dt = dt_op(
        dsl.InputArgumentPath(_transform.outputs['X_train']),
        dsl.InputArgumentPath(_transform.outputs['y_train']),
        dsl.InputArgumentPath(_transform.outputs['X_test']),
        str(dt_filename)+'.pkl',
        n_splits
    ).after(_transform)

    _knn = knn_op(
        dsl.InputArgumentPath(_transform.outputs['X_train']),
        dsl.InputArgumentPath(_transform.outputs['y_train']),
        dsl.InputArgumentPath(_transform.outputs['X_test']),
        n_neighbors,
        n_splits,
        str(knn_filename)+'.pkl',
    ).after(_transform)

    models = [
        dsl.InputArgumentPath(_svm.outputs['svm_model']),
        dsl.InputArgumentPath(_lr.outputs['lr_model']),
        dsl.InputArgumentPath(_dt.outputs['dt_model']),
        dsl.InputArgumentPath(_knn.outputs['knn_model']),
    ]
    _save_s3 = save_s3_op(
        models,
        location,
        [svm_filename, lr_filename, dt_filename, knn_filename]
    ).after(_svm,_lr,_dt,_knn).apply(aws.use_aws_secret(secret_name='s3-secrets'))

    _evaluation_knn = evaluation_op(
        dsl.InputArgumentPath(_knn.outputs['knn_predict']),
        dsl.InputArgumentPath(_transform.outputs['y_test']),
        [label1, label2, label3],
        dsl.InputArgumentPath(_transform.outputs['y_train']),
        dsl.InputArgumentPath(_knn.outputs['knn_y_scores'])
    ).after(_knn)
    _evaluation_dt = evaluation_op(
        dsl.InputArgumentPath(_dt.outputs['dt_predict']),
        dsl.InputArgumentPath(_transform.outputs['y_test']),
        [label1, label2, label3],
        dsl.InputArgumentPath(_transform.outputs['y_train']),
        dsl.InputArgumentPath(_dt.outputs['dt_y_scores'])
    ).after(_dt)
    _evaluation_svm = evaluation_op(
        dsl.InputArgumentPath(_svm.outputs['svm_predict']),
        dsl.InputArgumentPath(_transform.outputs['y_test']),
        [label1, label2, label3],
        dsl.InputArgumentPath(_transform.outputs['y_train']),
        dsl.InputArgumentPath(_dt.outputs['svm_y_scores'])
    ).after(_svm)
    _evaluation_svm = evaluation_op(
        dsl.InputArgumentPath(_lr.outputs['lr_predict']),
        dsl.InputArgumentPath(_transform.outputs['y_test']),
        [label1, label2, label3],
        dsl.InputArgumentPath(_transform.outputs['y_train']),
        dsl.InputArgumentPath(_dt.outputs['lr_y_scores'])
    ).after(_lr)


if __name__ == '__main__':
    import kfp.compiler as compiler

    compiler.Compiler().compile(iris_train_pipeline, __file__ + '.tar.gz')