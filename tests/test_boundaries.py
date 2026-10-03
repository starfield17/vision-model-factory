"""Tests for architectural boundary checking.

The rule this suite protects: a boundary check that has never failed is not known to work.
Every forbidden form the checker claims to catch gets a probe, because the forms that were
missing the first time (relative imports, dynamic imports) were precisely the ones the
absolute-import probe could not see.
"""

from pathlib import Path

import pytest

from vision_model_factory.check_boundaries import (
    check_boundaries,
    get_imported_modules,
    module_name_for,
    resolve_import_target,
)

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
PKG_DIR = SRC_DIR / "vision_model_factory"
PROBE_RELATIVE = Path("contracts") / "_probe_forbidden_import.py"


@pytest.fixture
def probe():
    """Write a throwaway module into `contracts/`, then remove it.

    Placed in `contracts` because that layer is forbidden from importing everything else,
    so any real import of a sibling layer is a violation by definition.
    """
    path = PKG_DIR / PROBE_RELATIVE
    assert not path.exists(), "a leftover probe would fail unrelated boundary tests"

    def write(source: str):
        path.write_text(source, encoding="utf-8")
        try:
            return check_boundaries(SRC_DIR)
        finally:
            path.unlink()

    yield write

    assert not path.exists()


def test_clean_repository_boundaries():
    assert check_boundaries(SRC_DIR) == []


def test_absolute_forbidden_import_is_detected(probe):
    violations = probe(
        "from vision_model_factory.trainers import BaseTrainerAdapter\n"
    )
    assert any("LAYER FORBIDDEN" in v for v in violations), violations


def test_relative_import_is_detected(probe):
    """`from ..trainers import X` resolved to nothing at all in the first implementation."""
    violations = probe("from ..trainers import BaseTrainerAdapter\n")
    assert any("LAYER FORBIDDEN" in v for v in violations), violations


def test_relative_import_without_module_is_detected(probe):
    """`from .. import trainers` names a submodule, not a module path."""
    violations = probe("from .. import trainers\n")
    assert any("LAYER FORBIDDEN" in v for v in violations), violations


def test_relative_import_of_a_forbidden_submodule_is_detected(probe):
    """`from ..evaluation.metrics import x` names a module two levels deep."""
    violations = probe("from ..evaluation.metrics import compute_ap\n")
    assert any("LAYER FORBIDDEN" in v for v in violations), violations


def test_importlib_dynamic_import_is_detected(probe):
    violations = probe('import importlib\nimportlib.import_module("vision_model_factory.trainers")\n')
    assert any("LAYER FORBIDDEN" in v for v in violations), violations


def test_dunder_import_is_detected(probe):
    violations = probe('__import__("vision_model_factory.release.publisher")\n')
    assert any("LAYER FORBIDDEN" in v for v in violations), violations


def test_submodule_dynamic_import_is_detected(probe):
    violations = probe(
        'from importlib import import_module\nimport_module("vision_model_factory.evaluation.metrics")\n'
    )
    assert any("LAYER FORBIDDEN" in v for v in violations), violations


def test_cross_repository_import_is_detected(probe):
    for source, marker in (
        ("import vision_runtime\n", "CROSS-REPO FORBIDDEN"),
        ("import vision_runtime.decoder\n", "CROSS-REPO FORBIDDEN"),
        ("from vision_data_factory.contracts import schemas\n", "CROSS-REPO FORBIDDEN"),
        ("from backend.app.services import x\n", "CROSS-REPO FORBIDDEN"),
        ('import_module("vision_runtime.stuff")\n', "CROSS-REPO FORBIDDEN"),
    ):
        violations = probe(source)
        assert any(marker in v for v in violations), (source, violations)


def test_cross_layer_directions_are_all_covered(probe):
    """Each declared forbidden direction gets its own probe.

    A single probe only proves the checker works for the pair it happened to test; the
    rules are a table, so the tests are a table too.
    """
    cases = [
        ("contracts", "trainers"), ("contracts", "evaluation"), ("contracts", "export"),
        ("contracts", "experiments"), ("contracts", "release"), ("contracts", "cli"),
        ("trainers", "experiments"), ("trainers", "release"),
        ("evaluation", "experiments"), ("evaluation", "release"),
        ("export", "trainers"), ("export", "experiments"), ("export", "release"),
        ("experiments", "release"),
    ]
    for layer, forbidden in cases:
        probe_file = PKG_DIR / layer / "_probe_forbidden_import.py"
        try:
            probe_file.write_text(
                f"from vision_model_factory.{forbidden} import anything\n", encoding="utf-8"
            )
            violations = check_boundaries(SRC_DIR)
            assert any("LAYER FORBIDDEN" in v for v in violations), f"{layer} -> {forbidden} not caught"
        finally:
            probe_file.unlink()


def test_allowed_imports_are_not_flagged(probe):
    """Reverse the check: a legal import must produce no violation, or the checker just
    reports everything and the earlier assertions mean nothing."""
    assert probe("from vision_model_factory.contracts.models import ClassMapItem\n") == []
    assert probe("import numpy\nimport json\nfrom pathlib import Path\n") == []
    assert probe("from .hashing import compute_sha256_file\n") == []
    assert probe("from __future__ import annotations\n") == []


def test_repository_is_clean_after_all_probes():
    assert check_boundaries(SRC_DIR) == []


def test_relative_resolution_matches_python_semantics():
    """Unit-level checks on the resolver, so a wrong level calculation is visible directly."""
    # Level 1 from a module inside a package resolves to that package.
    assert resolve_import_target("models", 1, "vision_model_factory.contracts.validators", False) == (
        "vision_model_factory.contracts.models"
    )
    # Level 2 climbs one more level.
    assert resolve_import_target("trainers", 2, "vision_model_factory.contracts.validators", False) == (
        "vision_model_factory.trainers"
    )
    # From a package __init__, level 1 is the package itself.
    assert resolve_import_target("registry", 1, "vision_model_factory.contracts", True) == (
        "vision_model_factory.contracts.registry"
    )
    # Climbing past the package root cannot name an internal layer.
    assert resolve_import_target("thing", 9, "vision_model_factory.contracts.validators", False) == ""


def test_module_name_for_modules_and_packages():
    assert module_name_for(PKG_DIR / "contracts" / "models.py", PKG_DIR) == (
        "vision_model_factory.contracts.models", False
    )
    assert module_name_for(PKG_DIR / "contracts" / "__init__.py", PKG_DIR) == (
        "vision_model_factory.contracts", True
    )


def test_relative_import_escaping_the_package_root_names_no_layer(probe):
    """`from ...x import y` from `contracts/` would be an ImportError at runtime.

    The resolver returns an empty target for it, which is correct: it cannot name an
    internal layer, so it must not be reported as one. Pinned here because "returns
    nothing" is also what a broken resolver looks like.
    """
    assert resolve_import_target("whatever", 3, "vision_model_factory.contracts.probe", False) == ""
    # And the probe itself produces no layer violation (it is still invalid Python to run,
    # but boundary checking is not a syntax/ImportError checker).
    violations = probe("from ...anything import x\n")
    assert not any("LAYER FORBIDDEN" in v for v in violations), violations


def test_import_inventory_of_a_real_module(tmp_path: Path):
    """`get_imported_modules` sees every form at once, with correct line numbers."""
    sample = tmp_path / "vision_model_factory" / "contracts" / "sample_probe.py"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text(
        "import numpy\n"                                  # 1
        "from vision_model_factory.contracts import hashing\n"  # 2
        "from ..export import decoder\n"                  # 3
        "import importlib\n"                              # 4
        "importlib.import_module('vision_model_factory.trainers')\n",  # 5
        encoding="utf-8",
    )
    fake_pkg = tmp_path / "vision_model_factory"
    found = {line: name for line, name in get_imported_modules(sample, fake_pkg)}
    assert found[1] == "numpy"
    assert found[2] == "vision_model_factory.contracts.hashing"
    assert found[3] == "vision_model_factory.export.decoder"
    assert found[5] == "vision_model_factory.trainers"
    sample.unlink()


def test_missing_package_is_reported_not_silently_ok(tmp_path: Path):
    violations = check_boundaries(tmp_path / "nowhere")
    assert violations and "Directory not found" in violations[0]


def test_unparseable_module_is_reported(tmp_path: Path):
    """A file that cannot be parsed must fail the check rather than pass it silently."""
    broken = tmp_path / "vision_model_factory" / "contracts" / "broken_probe.py"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("def oops(:\n", encoding="utf-8")

    violations = check_boundaries(tmp_path)
    assert any("PARSE FAILED" in v for v in violations), violations
