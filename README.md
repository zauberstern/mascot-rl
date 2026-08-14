<p align="center">
  <img src="assets/logo.png" width="168" alt="MascotRL">
</p>
<h1 align="center">MascotRL</h1>
<p align="center">
  Can reinforcement learning produce auditable trading styles<br>
  from option-implied features?
</p>
<p align="center">
  <a href="https://github.com/zauberstern/mascot-rl/actions/workflows/ci.yml">
    <img src="https://github.com/zauberstern/mascot-rl/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
</p>

## What this is

MascotRL studies sequential allocation of a unit budget across equity names. Each day a policy maps observations to simplex weights (long-only in the headline arm). Signals come from option-implied surface features; the research arms are equity-only, options-informed, and a mixed blend. No options are traded in the evaluation estimand.

The research question has two parts. First, can a broad spectrum of RL allocators clear after-cost peer comparisons under combinatorial purged cross-validation (CPCV)? Second, do the resulting policies show trading styles you can audit from realised weight paths, not from the architecture label on the training cell? The spectrum spans seven algorithms, ten objectives, four network bodies, and four weight-head families (sparse tilt and softmax as primaries; tanh-L1 and entmax as secondaries). Six named archetypes (Cheetah, Fox, Tortoise, Magpie, Hummingbird, Owl) score designed versus observed behaviour from weights.

## How evaluation works

Training and evaluation share matched trading costs. CPCV builds many purged train/test splits with embargo gaps so label horizons cannot leak across folds; path-level Sharpe distributions feed overfitting audits (PBO, deflated Sharpe, path diagnostics). Nested walk-forward is a separate chronological confirmation pass. It is not CPCV and must not be treated as interchangeable.

A fingerprint of universe membership, friction parameters, rebalance cadence, and fold geometry must match before any strategy comparison runs. Soft fees present in evaluation but absent in training abort the run. Claim fields fail closed: this repository does not assert capital deployment or live tradability. Vendor microdata (OptionMetrics, CRSP, WRDS) is not redistributed here.

## What is in the tree

| Path | Role |
|------|------|
| `src/mascotrl/` | Library: data lake helpers, spectrum registry, CPCV eval, envs, policies |
| `config/` | Versioned spectrum YAML grids |
| `scripts/` | Campaign runners and ingest utilities |
| `data/pseudo/` | Synthetic panel for offline smoke tests |
| `tests/` | Pytest suite |
| `deploy/` | AWS Batch burst templates (sanitized for public CI) |

## Install

```bash
git clone https://github.com/zauberstern/mascot-rl.git
cd mascot-rl
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# optional C++ pricing engine (OpenCL host required)
python setup_extensions.py
```

Copy `.env.example` to `.env` for local paths and credentials. Never commit `.env`.

## Example

```bash
python examples/pseudo_cpcv_smoke.py
```

Loads the pseudo constituent panel and prints CPCV fold and path counts for the default geometry.

## Tests

```bash
pytest -q -m "not slow"
make test   # same, via Makefile
make lint   # ruff crash rules on src/ and tests/
```

CI runs on Python 3.12 with coverage and a test-count floor.

## License

MIT. See [LICENSE](LICENSE). If you find credentials or other secrets in the tree, report them privately to the maintainer; do not open a public issue with the material attached.
