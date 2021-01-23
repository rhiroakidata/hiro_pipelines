import argparse, boto3, joblib
import numpy as np

from utils.s3 import get_s3_configmap

def load_s3(location, model_name):
    s3 = boto3.resource('s3')
    bucket, region = get_s3_configmap()

    output_file = location + model_name

    with open(output_file, 'wb') as data:
        s3.Bucket(bucket).download_fileobj(output_file, data)

    model = joblib.load(model_name)

    np.save("model.npy", model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--location')
    parser.add_argument('--model_name')

    args = parser.parse_args()
    load_s3(args.location, args.model_name)