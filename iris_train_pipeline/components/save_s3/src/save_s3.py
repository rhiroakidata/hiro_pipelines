import argparse
import joblib
import boto3
import numpy as np

from utils.s3 import get_s3_configmap

def save_s3(model, location, model_filename):
    bucket, region = get_s3_configmap()
    s3 = boto3.resource('s3')
    output_file = location + model_filename
    model_converted = joblib.dump(model, model_filename)

    print(f'saving model {model_converted}...')

    s3.Bucket(bucket).put_object(Key=output_file, Body=model_converted)

    np.save('model.npy',model_converted)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--location')
    parser.add_argument('--model_filename')
    args = parser.parse_args()
    save_s3(args.model)