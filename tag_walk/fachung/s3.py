from luigi.contrib.s3 import S3Client
from luigi.contrib.s3 import S3Target

import fachung.configuration as conf

def get_s3_client(self):
    access_key = conf.get_config().get('s3', 'aws_access_key_id')
    secret = conf.get_config().get('s3', 'aws_secret_access_key')

    return S3Client(access_key, secret)
