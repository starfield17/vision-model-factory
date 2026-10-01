"""Tests for architectural boundary checking."""

from pathlib import Path

from vision_model_factory.check_boundaries import check_boundaries


def test_clean_repository_boundaries():
    """Verify that current codebase has zero architectural boundary violations."""
    src_dir = Path(__file__).resolve().parent.parent / "src"
    violations = check_boundaries(src_dir)
    assert violations == [], f"Unexpected boundary violations: {violations}"


def test_intentional_boundary_violation_failure():
    """
    Surveyor verification requirement:
    Intentionally write a forbidden import, run the check, watch it fail, then delete it.
    A check that has never failed is not known to work.
    """
    src_dir = Path(__file__).resolve().parent.parent / "src"
    contracts_dir = src_dir / "vision_model_factory" / "contracts"
    probe_file = contracts_dir / "_probe_forbidden_import.py"

    # Test 1: Forbidden internal layer import (contracts importing trainers)
    try:
        probe_file.write_text(
            "# Intentional forbidden import for test\n"
            "from vision_model_factory.trainers import BaseTrainerAdapter\n",
            encoding="utf-8",
        )
        violations = check_boundaries(src_dir)
        assert len(violations) > 0, "Boundary check failed to detect forbidden internal layer import!"
        assert any("LAYER FORBIDDEN" in v for v in violations)
    finally:
        if probe_file.exists():
            probe_file.unlink()

    # Test 2: Forbidden cross-repository import (importing vision_runtime or backend.app)
    try:
        probe_file.write_text(
            "# Intentional forbidden cross-repo import\n"
            "import vision_runtime\n",
            encoding="utf-8",
        )
        violations = check_boundaries(src_dir)
        assert len(violations) > 0, "Boundary check failed to detect forbidden cross-repo import!"
        assert any("CROSS-REPO FORBIDDEN" in v for v in violations)
    finally:
        if probe_file.exists():
            probe_file.unlink()

    # Verify repository is clean again
    clean_violations = check_boundaries(src_dir)
    assert clean_violations == []
