# AWS-4: Burst training image. Pin from deploy/training_remote/requirements.lock.
FROM python:3.12-slim-bookworm
WORKDIR /app
COPY deploy/training_remote/requirements.lock /app/requirements.lock
# torch==*+cpu is published on the PyTorch CPU index, not on PyPI.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /app/requirements.lock
# Build + runtime libs for Layer-1/2 pricing extensions (bookworm glibc).
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    libgomp1 \
    libomp-dev \
    ocl-icd-libopencl1 \
    ocl-icd-opencl-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*
COPY . /app
RUN chmod +x /app/deploy/aws_burst/docker/entrypoint.sh
ENV MASCOTRL_REQUIREMENTS_LOCK=/app/requirements.lock
ENV PYTHONPATH=/app
# Bake lock sha for provenance.
RUN python - <<'PY'
from pathlib import Path
import hashlib
p = Path("/app/requirements.lock")
h = hashlib.sha256(p.read_bytes()).hexdigest()
Path("/app/REQUIREMENTS_LOCK_SHA256").write_text(h + "\n")
print(h)
PY
# Layer-1 (cpp_rbergomi) + Layer-2 (polaris) against image glibc.
RUN python setup_extensions.py \
    && python - <<'PY'
from pathlib import Path

import cpp_rbergomi
from src.pricing.interface import polaris_pricer_cpp
from src.spectrum.cell_schema import validate_cell_cfg
from src.spectrum.yaml_loader import load_cell_yaml

assert cpp_rbergomi is not None, "cpp_rbergomi missing from image"
assert polaris_pricer_cpp is not None, "polaris_pricer_cpp missing from image"
w = cpp_rbergomi.WorldConfig()
assert hasattr(w, "heston_scheme"), "WorldConfig missing heston_scheme (QE-M pin)"
assert int(w.heston_scheme) == 2, f"heston_scheme default must be QE-M=2, got {w.heston_scheme}"
# Fail the image if any HYBRID/FEATNET cell YAML drifts from SCHEMA (caught
# the heston_scheme unknown-key Burst failure before submit).
for root in (
    Path("/app/config/spectrum/cherrypick_hybrid"),
    Path("/app/config/spectrum/cherrypick_featnet"),
    Path("/app/config/spectrum/cherrypick_regime"),
):
    if not root.is_dir():
        continue
    for path in sorted(root.glob("*.yaml")):
        validate_cell_cfg(load_cell_yaml(path), path=str(path))
print("cpp_rbergomi_ok", getattr(cpp_rbergomi, "__file__", cpp_rbergomi))
print("polaris_ok", getattr(polaris_pricer_cpp, "__file__", polaris_pricer_cpp))
print("hybrid_featnet_schema_ok")
PY
ENTRYPOINT ["/app/deploy/aws_burst/docker/entrypoint.sh"]
CMD ["python", "deploy/aws_burst/docker/cell_runner.py"]
