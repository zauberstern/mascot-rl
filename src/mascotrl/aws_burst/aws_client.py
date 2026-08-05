"""Thin boto3 client for AWS burst operations."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# boto3 standard retry: exponential backoff with full jitter
# (docs.aws.amazon.com/boto3/latest/guide/retries.html).
_RETRY = Config(
    retries={"mode": "standard", "max_attempts": 5},
    connect_timeout=10,
    read_timeout=60,
)


class BurstClient:
    def __init__(self, profile: str, region: str) -> None:
        self.profile = profile
        self.region = region
        self._session = boto3.Session(profile_name=profile, region_name=region)
        self._clients: dict[str, Any] = {}

    def _client(self, service: str):
        if service not in self._clients:
            self._clients[service] = self._session.client(service, config=_RETRY)
        return self._clients[service]

    def _s3(self):
        return self._client("s3")

    def _batch(self):
        return self._client("batch")

    def _ecr(self):
        return self._client("ecr")

    def _cfn(self):
        return self._client("cloudformation")

    def _sts(self):
        return self._client("sts")

    def _budgets(self):
        # Budgets is a global endpoint (us-east-1).
        if "budgets" not in self._clients:
            self._clients["budgets"] = self._session.client(
                "budgets", region_name="us-east-1", config=_RETRY
            )
        return self._clients["budgets"]

    def account_id(self) -> str:
        return str(self._sts().get_caller_identity()["Account"])

    def ensure_bucket(self, bucket: str) -> None:
        try:
            self._s3().head_bucket(Bucket=bucket)
            return
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            http = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
            # Only create when the bucket is missing; re-raise other errors.
            if code not in {"404", "NoSuchBucket", "NotFound"} and http != 404:
                raise
        params: dict[str, Any] = {"Bucket": bucket}
        if self.region != "us-east-1":
            params["CreateBucketConfiguration"] = {"LocationConstraint": self.region}
        self._s3().create_bucket(**params)

    def put_json(self, bucket: str, key: str, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        self._s3().put_object(
            Bucket=bucket,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )

    def get_json(self, bucket: str, key: str) -> dict[str, Any]:
        obj = self._s3().get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))

    def put_file_with_sha(self, bucket: str, key: str, path: Path) -> str:
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        # Do not hash .sha256 sidecars (avoids .sha256.sha256 litter).
        if str(key).endswith(".sha256"):
            self._s3().put_object(Bucket=bucket, Key=key, Body=data)
            return digest
        sha_key = f"{key}.sha256"
        # Skip rewrite when remote sha already matches (panel bundles are ~400MB).
        try:
            remote = (
                self._s3()
                .get_object(Bucket=bucket, Key=sha_key)["Body"]
                .read()
                .decode("utf-8")
                .strip()
            )
            if remote == digest and self.head_exists(bucket, key):
                return digest
        except ClientError:
            pass
        self._s3().put_object(Bucket=bucket, Key=key, Body=data)
        self._s3().put_object(
            Bucket=bucket,
            Key=sha_key,
            Body=(digest + "\n").encode("utf-8"),
        )
        return digest

    def list_keys(self, bucket: str, prefix: str) -> list[str]:
        keys: list[str] = []
        paginator = self._s3().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents") or []:
                keys.append(str(item["Key"]))
        return keys

    def head_exists(self, bucket: str, key: str) -> bool:
        try:
            self._s3().head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False

    def download_prefix(
        self,
        bucket: str,
        prefix: str,
        dest: Path,
        *,
        skip_rel_prefixes: tuple[str, ...] = (),
    ) -> list[Path]:
        dest.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        skips = tuple(str(s).lstrip("/") for s in skip_rel_prefixes if str(s).strip())
        for key in self.list_keys(bucket, prefix):
            rel = key[len(prefix) :].lstrip("/")
            if not rel:
                continue
            if skips and any(
                rel == s or rel.startswith(s if s.endswith("/") else f"{s}/")
                for s in skips
            ):
                continue
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            self._s3().download_file(bucket, key, str(out))
            written.append(out)
        return written

    def submit_array_job(
        self,
        *,
        job_name: str,
        job_queue: str,
        job_definition: str,
        array_size: int,
        client_token: str,
        container_overrides: dict[str, Any] | None = None,
        attempt_duration_seconds: int | None = None,
    ) -> dict[str, str]:
        # Batch SubmitJob has no ClientToken in the public botocore model
        # (verified boto3 1.43). Idempotency is via a deterministic jobName
        # suffix derived from client_token (shard-manifest sha).
        token_suffix = "".join(ch for ch in client_token if ch.isalnum())[:8]
        safe_name = f"{job_name}-{token_suffix}" if token_suffix else job_name
        # AWS jobName max length is 128.
        safe_name = safe_name[:128]
        n = int(array_size)
        if n < 1:
            raise ValueError(f"array_size_must_be_positive: got {n}")
        params: dict[str, Any] = {
            "jobName": safe_name,
            "jobQueue": job_queue,
            "jobDefinition": job_definition,
        }
        # AWS Batch rejects array size 1; submit a plain job and inject index 0
        # so cell_runner's AWS_BATCH_JOB_ARRAY_INDEX contract still holds.
        overrides = dict(container_overrides or {})
        if n == 1:
            # Batch rejects array size 1 and strips AWS_BATCH_* overrides on
            # plain jobs. Pass MASCOTRL_ARRAY_INDEX; entrypoint/cell_runner
            # map it onto AWS_BATCH_JOB_ARRAY_INDEX.
            env = list(overrides.get("environment") or [])
            if not any(e.get("name") == "MASCOTRL_ARRAY_INDEX" for e in env):
                env.append({"name": "MASCOTRL_ARRAY_INDEX", "value": "0"})
            overrides["environment"] = env
        else:
            params["arrayProperties"] = {"size": n}
        if overrides:
            params["containerOverrides"] = overrides
        if attempt_duration_seconds is not None:
            params["timeout"] = {"attemptDurationSeconds": int(attempt_duration_seconds)}
        resp = self._batch().submit_job(**params)
        return {
            "jobId": str(resp["jobId"]),
            "jobArn": str(resp.get("jobArn") or resp["jobId"]),
            "jobName": safe_name,
        }

    def describe_jobs(self, job_ids: list[str]) -> list[dict[str, Any]]:
        if not job_ids:
            return []
        resp = self._batch().describe_jobs(jobs=list(job_ids))
        return list(resp.get("jobs") or [])

    def list_jobs(self, job_queue: str, status: str) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        paginator = self._batch().get_paginator("list_jobs")
        for page in paginator.paginate(jobQueue=job_queue, jobStatus=status):
            jobs.extend(page.get("jobSummaryList") or [])
        return jobs

    def wait_for_array(
        self,
        job_id: str,
        *,
        poll_seconds: float = 15.0,
        timeout_seconds: float = 14400.0,
    ) -> dict[str, Any]:
        """Poll describe_jobs until the array parent reaches a terminal status."""
        terminal = {"SUCCEEDED", "FAILED"}
        deadline = time.monotonic() + float(timeout_seconds)
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            jobs = self.describe_jobs([job_id])
            if not jobs:
                raise RuntimeError(f"batch_job_missing: {job_id}")
            last = jobs[0]
            status = str(last.get("status") or "")
            if status in terminal:
                return last
            time.sleep(float(poll_seconds))
        raise TimeoutError(
            f"batch_wait_timeout: job_id={job_id} last_status={last.get('status')}"
        )

    def describe_compute_environments(
        self, names: list[str] | None = None
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {}
        if names:
            kwargs["computeEnvironments"] = list(names)
        resp = self._batch().describe_compute_environments(**kwargs)
        return list(resp.get("computeEnvironments") or [])

    def register_job_definition(self, name: str, props: dict[str, Any]) -> str:
        resp = self._batch().register_job_definition(jobDefinitionName=name, **props)
        return str(resp["jobDefinitionArn"])

    def ecr_ensure_repo(self, repo_name: str) -> str:
        try:
            resp = self._ecr().describe_repositories(repositoryNames=[repo_name])
            return str(resp["repositories"][0]["repositoryUri"])
        except self._ecr().exceptions.RepositoryNotFoundException:
            resp = self._ecr().create_repository(repositoryName=repo_name)
            return str(resp["repository"]["repositoryUri"])

    def ecr_image_digest(self, repo_name: str, image_tag: str = "latest") -> str:
        resp = self._ecr().describe_images(
            repositoryName=repo_name,
            imageIds=[{"imageTag": image_tag}],
        )
        details = resp["imageDetails"][0]
        for digest in details.get("imageDigest", "").split(","):
            if digest.startswith("sha256:"):
                return digest
        return str(details.get("imageDigest") or "")

    def deploy_stack(
        self,
        stack_name: str,
        template_path: Path,
        parameters: list[dict[str, str]] | None = None,
    ) -> None:
        template = template_path.read_text(encoding="utf-8")
        kwargs: dict[str, Any] = {
            "StackName": stack_name,
            "TemplateBody": template,
            "Capabilities": ["CAPABILITY_NAMED_IAM"],
        }
        if parameters:
            kwargs["Parameters"] = parameters
        cfn = self._cfn()
        created = False
        try:
            cfn.create_stack(**kwargs)
            created = True
        except cfn.exceptions.AlreadyExistsException:
            cfn.update_stack(**kwargs)
        waiter_name = "stack_create_complete" if created else "stack_update_complete"
        cfn.get_waiter(waiter_name).wait(StackName=stack_name)
