#!/bin/bash

AWS_PROFILE="fachung"

tar -zcvf ./tmp/fachung.tar.gz \
	--exclude=*.pyc \
	--exclude=*..egg-info \
	--exclude=./data \
	--exclude=.ropeproject .

aws s3 cp ./tmp/fachung.tar.gz s3://fachung/archives/ --profile="$AWS_PROFILE"

rm ./tmp/fachung.tar.gz
