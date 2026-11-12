import ast
from pathlib import Path

import pytest

from compute_maintenance.domain.snapshot import FrozenMap


def test_planners_have_no_side_effect_adapter_dependencies():
    root = Path(__file__).resolve().parents[1] / "src/compute_maintenance/domain"
    # rc12: balance.py became the package domain/balance/ (every module is checked).
    planners = sorted(p.relative_to(root).as_posix() for p in (root / "balance").glob("*.py"))
    assert len(planners) >= 7, planners
    for name in (*planners, "drain.py", "policies.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not any(".adapters" in module or ".application" in module or
                           module.split('.')[0] in ("requests", "openstack", "subprocess", "time", "os")
                           for module in modules), (name, modules)


def test_snapshot_nested_inputs_cannot_be_changed_by_planner():
    source = {"host": {"used": 1}}
    frozen = FrozenMap(source)
    source["host"]["used"] = 2
    assert frozen["host"]["used"] == 1
    with pytest.raises(TypeError):
        frozen["host"]["used"] = 3
