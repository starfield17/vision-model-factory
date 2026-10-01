"""Architectural boundary checker for Vision Model Factory."""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Layer forbidden import rules: source layer -> set of forbidden target subpackages/modules
FORBIDDEN_INTERNAL_IMPORTS: Dict[str, Set[str]] = {
    "contracts": {
        "trainers",
        "evaluation",
        "export",
        "experiments",
        "release",
        "cli",
    },
    "trainers": {
        "experiments",
        "release",
        "cli",
    },
    "evaluation": {
        "experiments",
        "release",
        "cli",
    },
    "export": {
        "experiments",
        "trainers",
        "release",
        "cli",
    },
    "experiments": {
        "release",
        "cli",
    },
    "release": {
        "cli",
    },
}

# Cross-repository forbidden package names
FORBIDDEN_EXTERNAL_PACKAGES: Set[str] = {
    "vision_runtime",
    "vision_data_factory",
    "backend.app",
    "app.datasets",
    "app.review",
    "app.services",
}


def get_imported_modules(py_file: Path) -> List[Tuple[int, str]]:
    """Parse a Python source file and return (line_number, imported_module_name)."""
    with py_file.open("r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=str(py_file))
        except SyntaxError as e:
            return [(1, f"SYNTAX_ERROR: {e}")]

    imports: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append((node.lineno, node.module))
    return imports


def check_boundaries(src_dir: Path) -> List[str]:
    """
    Inspect all Python files in src_dir and report all boundary violations.
    """
    violations: List[str] = []
    src_dir = src_dir.resolve()
    base_pkg_dir = src_dir / "vision_model_factory"

    if not base_pkg_dir.is_dir():
        violations.append(f"Directory not found: {base_pkg_dir}")
        return violations

    for py_file in base_pkg_dir.rglob("*.py"):
        rel_path = py_file.relative_to(base_pkg_dir)
        parts = rel_path.parts

        layer = parts[0] if len(parts) > 1 else "root"
        imports = get_imported_modules(py_file)

        for lineno, mod_name in imports:
            # 1. Check cross-repository forbidden packages
            for forbidden_ext in FORBIDDEN_EXTERNAL_PACKAGES:
                if mod_name == forbidden_ext or mod_name.startswith(f"{forbidden_ext}."):
                    msg = f"[CROSS-REPO FORBIDDEN] {py_file}:{lineno} imports '{mod_name}'"
                    violations.append(msg)

            # 2. Check internal layer restrictions
            if layer in FORBIDDEN_INTERNAL_IMPORTS:
                forbidden_layers = FORBIDDEN_INTERNAL_IMPORTS[layer]
                for f_layer in forbidden_layers:
                    pattern_abs = f"vision_model_factory.{f_layer}"
                    if mod_name == pattern_abs or mod_name.startswith(f"{pattern_abs}."):
                        msg = f"[LAYER FORBIDDEN] {py_file}:{lineno} in '{layer}' imports forbidden '{mod_name}'"
                        violations.append(msg)

    return violations


def main() -> None:
    src_path = Path(__file__).resolve().parent.parent
    violations = check_boundaries(src_path)

    if violations:
        print(f"FAILED: Found {len(violations)} architectural boundary violation(s):", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        sys.exit(1)
    else:
        print("OK: All architectural boundary checks passed successfully.")


if __name__ == "__main__":
    main()
