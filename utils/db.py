import base64

from kubernetes import client, config
from kubernetes.client.rest import ApiException

config.load_kube_config()
v1 = client.CoreV1Api()

def get_db_secrets():
    try:
        secret = v1.read_namespaced_secret("db-creds", "default")
        data = secret.data

        username = base64.b64decode(data['username'])
        password = base64.b64decode(data['password'])
        return username, password

    except ApiException as e:
        print("Exception when calling CoreV1Api->read_namespaced_secret: %s\n" % e)

def get_db_configmap():
    try:
        configmap = v1.read_namespaced_config_map("db-configmap")
        data = configmap.data
        host = base64.b64decode(data["host"])
        port = base64.b64decode(data["port"])
        bd = base64.b64decode(data["bd"])

        return bd, host, port

    except ApiException as e:
        print("Exception when calling CoreV1Api->read_namespaced_config_map: %s\n" % e)