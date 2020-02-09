#!/usr/bin/env bash

BUCKET=$1

if [ -z "$BUCKET" ]
then
      echo "BUCKET isn't specified; please check s3 for the name."
      exit 1
fi

S3_DIR="s3://$BUCKET/projects/2019-vdb"
DATE=`date '+%Y-%m-%d'`
echo "Back up directories for $DATE"

for path in `cat ./datasets/artifact-directories.txt`
do
    base_path=`basename "$path"`
    s3_path="$S3_DIR/$DATE/$base_path"
    echo "Backing up $path ➡ $s3_path"
    aws s3 sync $path $s3_path
done