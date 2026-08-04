"""YAML ``k`` must be read for tracked honesty on crucible workflows."""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from src.eval.yaml_honesty import TrackingDict, assert_yaml_honesty_tracked, load_workflow_keys
ROOT = Path(__file__).resolve().parents[1]
CRUCIBLE_YAML = ROOT / 'config' / 'workflows' / 'eq_alloc_crucible_k100.yaml'

def test_resolve_campaign_k_prefers_cli_but_touches_yaml():
    from scripts.run_eq_alloc_campaign import resolve_campaign_k
    cfg = TrackingDict({'k': 100, 'max_pool': 511})
    args = argparse.Namespace(k=40)
    assert resolve_campaign_k(args, cfg) == 40
    assert 'k' in cfg._accessed

def test_resolve_campaign_k_falls_back_to_yaml():
    from scripts.run_eq_alloc_campaign import resolve_campaign_k
    cfg = TrackingDict({'k': 100})
    args = argparse.Namespace(k=None)
    assert resolve_campaign_k(args, cfg) == 100
    assert 'k' in cfg._accessed
