#!/bin/bash

# :hankey:
AWS_ACCESS=$(sed -n '1,/fachung/d;/\[/,$d;/^$/d;p' ~/.aws/credentials | cut -d"=" -f2 | head -n1)
AWS_SECRET=$(sed -n '1,/fachung/d;/\[/,$d;/^$/d;p' ~/.aws/credentials | cut -d"=" -f2 | tail -n1)

terraform2 "${1}" \
	-var-file="./terraform.tfvars" \
	-var aws_access_key_id="${AWS_ACCESS}" \
	-var aws_secret_access_key="${AWS_SECRET}"
