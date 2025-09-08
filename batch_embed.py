import os
import sys
import time
import datetime
from typing import Optional

from google import genai
from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions
from google.cloud import storage


def upload_to_gcs(local_path: str, bucket_name: str, object_name: str) -> str:
    client = storage.Client()
    bucket = client.lookup_bucket(bucket_name)
    if bucket is None:
        raise RuntimeError(
            f"GCS bucket not found: {bucket_name}. Create it first or set BATCH_BUCKET to an existing bucket."
        )
    blob = bucket.blob(object_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{object_name}"


def list_gcs_uris(prefix_uri: str) -> None:
    if not prefix_uri.startswith("gs://"):
        print(f"Output URI is not a GCS prefix: {prefix_uri}")
        return
    parts = prefix_uri.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    client = storage.Client()
    bucket = client.lookup_bucket(bucket_name)
    if bucket is None:
        print(f"Could not list outputs; bucket not found: {bucket_name}")
        return

    print("Output files:")
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith(".jsonl"):
            print(f"gs://{bucket_name}/{blob.name}")


def require_env(name: str, default: Optional[str] = None) -> str:
    val = os.getenv(name, default)
    if val is None or val == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def main() -> None:
    project_id = require_env("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

    local_jsonl = os.getenv(
        "LOCAL_JSONL",
        os.path.join(os.getcwd(), "head_content.jsonl"),
    )
    if not os.path.isabs(local_jsonl):
        local_jsonl = os.path.abspath(local_jsonl)

    if not os.path.exists(local_jsonl):
        print(f"Local JSONL not found: {local_jsonl}")
        sys.exit(1)

    bucket_name = require_env("BATCH_BUCKET")
    prefix = os.getenv("BATCH_PREFIX", "embedding_batches")

    ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    object_name = f"{prefix}/inputs/{ts}/head_content.jsonl"
    output_uri = f"gs://{bucket_name}/{prefix}/outputs/{ts}/"

    print(f"Uploading input to GCS bucket '{bucket_name}'...")
    gcs_src = upload_to_gcs(local_jsonl, bucket_name, object_name)
    print(f"Uploaded: {gcs_src}")

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        http_options=HttpOptions(api_version="v1"),
    )

    print("Creating batch embeddings job...")
    job = client.batches.create(
        model="text-embedding-005",
        src=gcs_src,
        config=CreateBatchJobConfig(dest=output_uri),
    )

    print(f"Job name: {job.name}")
    print(f"Job state: {job.state}")

    completed_states = {
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_FAILED,
        JobState.JOB_STATE_CANCELLED,
        JobState.JOB_STATE_PAUSED,
    }

    while job.state not in completed_states:
        time.sleep(30)
        job = client.batches.get(name=job.name)
        print(f"Job state: {job.state}")
        if job.state == JobState.JOB_STATE_FAILED:
            print(f"Error: {job.error}")
            break

    if job.state == JobState.JOB_STATE_SUCCEEDED:
        print("Batch job succeeded.")
        print(f"Output prefix: {output_uri}")
        list_gcs_uris(output_uri)
    else:
        print(f"Batch job finished with state: {job.state}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
