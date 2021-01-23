#!/bin/bash -e
image_name=aws_account_id.dkr.ecr.region.amazonaws.com/nat_pipelines/iris_components/load_s3 # Specify the image name here
image_tag=latest
full_image_name=${image_name}:${image_tag}

cd "$(dirname "$0")"
docker build -t "${full_image_name}" .
docker push "$full_image_name"

# Output the strict image name (which contains the sha256 image digest)
docker inspect --format="{{index .RepoDigests 0}}" "${full_image_name}"