from kfp import dsl
from kfp import aws

import yaml

# ========== Operations ============
def load_s3_op(model_name, location):
    return dsl.ContainerOp(
        name='Load Model from S3',
        image='rhiroakidata/iris_components/load_s3:latest',
        arguments=[
            '--model_name', model_name,
            '--location', location
        ],
        file_outputs={
            'iris_model': '/app/iris_model.npy',
        }
    )

# ========== PIPELINE ==============

@dsl.pipeline(
    name='Iris Prod Pipeline Example',
    description='Example with the Iris classification'
)
def iris_prod_pipeline(
    location: dsl.PipelineParam = dsl.PipelineParam(name='location', value='FOLDER_NAME_TO_MODELS'),
    model_name: dsl.PipelineParam = dsl.PipelineParam(name="model_name", value="MODEL NAME"),
    is_deploy: dsl.PipelineParam = dsl.PipelineParam(name="is_deploy", param_type='bool')
):
    _load_s3 = load_s3_op(
        location,
        model_name
    ).apply(aws.use_aws_secret(secret_name='s3-secrets'))

    seldon_config = yaml.load(open("iris_prod_pipeline/components/deploy/deploy_iris.yaml"))

    with dsl.Condition(is_deploy==True, name='deploy'):
        _deploy = dsl.ResourceOp(
            name="seldondeploy",
            k8s_resource=seldon_config,
            attribute_outputs={"name": "{.metadata.name}"})

        _deploy.after(_load_s3)



if __name__ == '__main__':
    import kfp.compiler as compiler

    compiler.Compiler().compile(iris_prod_pipeline, __file__ + '.tar.gz')