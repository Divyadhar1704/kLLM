"""Alert policies and dashboard (deploy/monitoring) only use metrics that exist: a renamed metric
would otherwise silently disable an alert."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from prometheus_client import REGISTRY

import app.core.metrics  # noqa: F401 - registers the metrics

MONITORING = Path(__file__).resolve().parents[2] / "deploy" / "monitoring"
METRIC_NAME = re.compile(r"\bkrishimitra_[a-z_]+")
SUFFIXES = ("_bucket", "_count", "_sum", "_total")


def known_metrics() -> set[str]:
    names: set[str] = set()
    for family in REGISTRY.collect():
        if family.name.startswith("krishimitra_"):
            names.update(sample.name for sample in family.samples)
            names.add(family.name)
    return names


def referenced(text: str) -> set[str]:
    return set(METRIC_NAME.findall(text))


def assert_known(names: set[str], where: str) -> None:
    known = known_metrics()
    for name in names:
        base = name
        for suffix in SUFFIXES:
            base = base.removesuffix(suffix)
        assert name in known or base in known, f"{where}: unknown metric {name}"


ALERT_FILES = sorted((MONITORING / "alerts").glob("*.yaml"))


def test_alert_files_exist() -> None:
    assert len(ALERT_FILES) >= 6


@pytest.mark.parametrize("path", ALERT_FILES, ids=lambda p: p.name)
def test_alert_policy(path: Path) -> None:
    policy = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert policy["displayName"].startswith("KrishiMitra:")
    assert policy["severity"] in ("CRITICAL", "ERROR", "WARNING")
    assert policy["documentation"]["content"].strip()
    queries = [c["conditionPrometheusQueryLanguage"]["query"] for c in policy["conditions"]]
    assert queries
    for query in queries:
        names = referenced(query)
        assert names, f"{path.name}: query uses no krishimitra metric"
        assert_known(names, path.name)


def test_dashboard() -> None:
    dashboard = json.loads((MONITORING / "dashboard.json").read_text(encoding="utf-8"))
    tiles = dashboard["mosaicLayout"]["tiles"]
    assert len(tiles) >= 6
    queries = [
        data_set["timeSeriesQuery"]["prometheusQuery"]
        for tile in tiles
        for data_set in tile["widget"]["xyChart"]["dataSets"]
    ]
    assert_known(set().union(*(referenced(q) for q in queries)), "dashboard.json")
