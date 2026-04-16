import os
import boto3

BUCKET = os.getenv("S3_BUCKET", "cloud-docs-1212")
PREFIX = os.getenv("S3_PREFIX", "llmops-docs/chunks/")
LOCAL_DIR = os.getenv("LOCAL_DATA_DIR", "rag/data")
REGION = os.getenv("AWS_REGION", "us-east-1")

os.makedirs(LOCAL_DIR, exist_ok=True)
s3 = boto3.client("s3", region_name=REGION)

paginator = s3.get_paginator("list_objects_v2")
count = 0
for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        if not key.endswith("docs_chunks.jsonl"):
            continue
        rel = key.replace(PREFIX, "", 1)
        dest = os.path.join(LOCAL_DIR, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        s3.download_file(BUCKET, key, dest)
        count += 1
        print("downloaded:", key)

print("total files:", count)
