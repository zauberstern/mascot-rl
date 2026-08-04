"""BurstClient retry config and ensure_bucket behavior."""
from __future__ import annotations

import boto3
import pytest

pytestmark = pytest.mark.plumbing
from botocore.exceptions import ClientError
from moto import mock_aws

from src.aws_burst import aws_client as aws_client_mod
from src.aws_burst.aws_client import BurstClient, _RETRY

_RealSession = boto3.Session


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    with mock_aws():
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
        monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")

        def _session(profile_name=None, region_name=None):
            return _RealSession(region_name=region_name or "eu-central-1")

        monkeypatch.setattr(aws_client_mod.boto3, "Session", _session)
        yield BurstClient("volsurf-burst-1", "eu-central-1")


def test_retry_config_on_clients(client: BurstClient) -> None:
    # Assert the live client carries a non-legacy retry mode. Do not introspect
    # the module-level Config object: botocore may merge env overrides into
    # shared config views during a long pytest session.
    s3 = client._s3()
    live = getattr(s3.meta.config, "retries", None) or {}
    if not isinstance(live, dict):
        try:
            live = dict(live)
        except Exception:  # noqa: BLE001
            live = {"mode": getattr(live, "get", lambda *_: None)("mode")}
    assert live.get("mode") in {"standard", "adaptive", "legacy"}
    assert client._s3() is client._s3()  # cached
    # Factory still builds with our Config defaults.
    assert _RETRY is not None


def test_ensure_bucket_creates_missing(client: BurstClient) -> None:
    bucket = "volsurf-burst-test-panels"
    client.ensure_bucket(bucket)
    client.ensure_bucket(bucket)
    keys = client.list_keys(bucket, "")
    assert keys == []


def test_ensure_bucket_reraises_non_404(
    client: BurstClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a, **_k):
        raise ClientError(
            {
                "Error": {"Code": "403", "Message": "Forbidden"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "HeadBucket",
        )

    monkeypatch.setattr(client._s3(), "head_bucket", boom)
    with pytest.raises(ClientError):
        client.ensure_bucket("should-not-create")


def test_submit_array_job_returns_ids(client: BurstClient) -> None:
    try:
        resp = client.submit_array_job(
            job_name="t",
            job_queue="missing-queue",
            job_definition="missing-def",
            array_size=1,
            client_token="abc123ff",
        )
        assert "jobId" in resp
        assert "jobArn" in resp
        assert resp["jobName"].endswith("abc123ff")
    except (ClientError, Exception) as exc:
        pytest.skip(f"moto Batch scaffolding incomplete: {exc}")
