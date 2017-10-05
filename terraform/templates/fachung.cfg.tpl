[project]
mode: local

[local]
base: /home/ubuntu/dev
project: tag_walk/tag_walk/
data: data/
outputs: outputs/
paperdoll: paperdoll/data/paperdoll_dataset.mat
asos: asos/data
fashionista: fashionista/fashionista-v0.2.2/compiled
tag_walk: tag_walk/

[asos_db]
host: localhost
port: 5432
user: gregoirelejay
password:
database: scrape

[s3]
aws_access_key_id: ${aws_access_key_id}
aws_secret_access_key: ${aws_secret_access_key}

[buckets]
fachung: fachung
