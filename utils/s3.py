import base64

from kubernetes import client, config
from kubernetes.client.rest import ApiException

config.load_kube_config()
v1 = client.CoreV1Api()

def get_s3_secrets():
    try:
        secret = v1.read_namespaced_secret("s3-secrets", "default")
        data = secret.data

        key_id = base64.b64decode(data['AWS_ACCESS_KEY_ID'])
        access_key = base64.b64decode(data['AWS_SECRET_ACCESS_KEY'])
        return key_id, access_key

    except ApiException as e:
        print("Exception when calling CoreV1Api->read_namespaced_secret: %s\n" % e)

def get_s3_configmap():
    try:
        configmap = v1.read_namespaced_config_map("s3-configmap")
        data = configmap.data
        bucket = base64.b64decode(data["S3_BUCKET"])
        region = base64.b64decode(data["S3_REGION"])

        return bucket, region

    except ApiException as e:
        print("Exception when calling CoreV1Api->read_namespaced_config_map: %s\n" % e)