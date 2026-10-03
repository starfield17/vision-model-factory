"""Architectural boundary checker.

Enforces the dependency direction declared in `AGENTS.md`. Everything the hard rules
promise is checked here, including the forms that silently evaded the first version of
this tool: relative imports (`from ..trainers import X`) and dynamic imports whose
module name is a string literal (`importlib.import_module("vision_model_factory.trainers")`
/ `__import__(...)`). Both were verified to pass the previous absolute-name-only scan.

Known, deliberate limits (stated rather than implied):
* Names built at runtime (`"vision_model_factory." + layer`, f-strings) are not tracked.
  Nothing in this repository does that; a change that starts to must be reviewed as a
  boundary change.
* Dynamic import through `getattr` chains and `exec` is not tracked.
* `from vision_model_factory import contracts` is visible because imported names are
  enumerated; a name bound some other way (a re-export under an unrelated name) is not.
* `scripts/` and `tests/` are not scanned: they are entry points and test fixtures, and
  the deliberate-violation probe writes into `src` on purpose.
"""

import ast
import sys
from pathlib import Path
from typing import List, Optional, Tuple

PACKAGE_NAME = "vision_model_factory"

# Layer forbidden import rules: layer -> set of layers it must never import.
FORBIDDEN_INTERNAL_IMPORTS = {
    "contracts": {"trainers", "evaluation", "export", "experiments", "release", "cli"},
    "trainers": {"experiments", "release", "cli"},
    "evaluation": {"experiments", "release", "cli"},
    "export": {"experiments", "trainers", "release", "cli"},
    "experiments": {"release", "cli"},
    "release": {"cli"},
}

# Cross-repository packages that must never be imported. Cross-repo communication happens
# only through serialized package artifacts.
FORBIDDEN_EXTERNAL_PACKAGES = {
    "vision_runtime",
    "vision_data_factory",
    "backend.app",
}

# Functions that import a module by name at runtime.
DYNAMIC_IMPORT_FUNCS = {"import_module", "__import__", "_import_"}


def resolve_import_target(
    module: Optional[str], level: int, importing_module: str, is_package: bool
) -> str:
    """Resolve the absolute module an `import from` statement refers to.

    `level == 0` is an absolute import. `level >= 1` resolves against the importing
    module's package using normal Python semantics: level 1 is the current package, each
    additional dot climbs one level. An empty result means the import escapes the
    package root and therefore cannot name an internal layer.
    """
    if level == 0:
        return module or ""

    parts = importing_module.split(".")
    base = parts[:] if is_package else parts[:-1]

    climb = level - 1
    if climb > 0:
        if climb >= len(base):
            return ""
        base = base[: len(base) - climb]

    if module:
        base = base + module.split(".")
    return ".".join(base)


def module_name_for(py_file: Path, base_pkg_dir: Path) -> Tuple[str, bool]:
    """Absolute module name of a file, and whether it is a package `__init__`."""
    rel = py_file.relative_to(base_pkg_dir.parent)
    parts = list(rel.parts)
    is_package = parts[-1] == "__init__.py"
    if is_package:
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts), is_package


def get_imported_modules(py_file: Path, base_pkg_dir: Path) -> List[Tuple[int, str]]:
    """Return (line_number, resolved_module_name) for every import this file can trigger."""
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    except SyntaxError as e:
        return [(1, f"SYNTAX_ERROR: {e}")]

    importing_module, is_package = module_name_for(py_file, base_pkg_dir)
    imports: List[Tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))

        elif isinstance(node, ast.ImportFrom):
            level = node.level or 0
            parent = (
                node.module if level == 0
                else resolve_import_target(node.module, level, importing_module, is_package)
            )
            if parent:
                imports.append((node.lineno, parent))

            # Enumerate the imported names too: `from x import y` may bind a submodule of
            # x, which the statement's own module name never mentions. Without this, the
            # absolute form `from vision_model_factory import trainers` passed while the
            # identical relative form `from .. import trainers` was caught.
            if parent:
                for alias in node.names:
                    if alias.name != "*":
                        imports.append((node.lineno, f"{parent}.{alias.name}"))

        elif isinstance(node, ast.Call):
            imports.extend(_dynamic_imports(node))

    return imports


def _dynamic_imports(node: ast.Call) -> List[Tuple[int, str]]:
    """Collect string-literal module names passed to runtime import functions."""
    func = node.func
    name = getattr(func, "attr", None) or getattr(func, "id", None)
    if name not in DYNAMIC_IMPORT_FUNCS or not node.args:
        return []
    first = node.args[0]
    if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
        return []
    return [(node.lineno, first.value)]


def layer_of(rel_path_parts: Tuple[str, ...]) -> str:
    return rel_path_parts[0] if len(rel_path_parts) > 1 else "root"


def check_boundaries(src_dir: Path) -> List[str]:
    """Inspect every module under `src_dir` and report all configured violations."""
    violations: List[str] = []
    src_dir = Path(src_dir).resolve()
    base_pkg_dir = src_dir / PACKAGE_NAME

    if not base_pkg_dir.is_dir():
        return [f"Directory not found: {base_pkg_dir}"]

    for py_file in sorted(base_pkg_dir.rglob("*.py")):
        rel_path = py_file.relative_to(base_pkg_dir)
        layer = layer_of(rel_path.parts)

        for lineno, mod_name in get_imported_modules(py_file, base_pkg_dir):
            if str(mod_name).startswith("SYNTAX_ERROR"):
                violations.append(f"[PARSE FAILED] {py_file}:{lineno} {mod_name}")
                continue

            for forbidden_ext in sorted(FORBIDDEN_EXTERNAL_PACKAGES):
                if mod_name == forbidden_ext or mod_name.startswith(f"{forbidden_ext}."):
                    violations.append(
                        f"[CROSS-REPO FORBIDDEN] {py_file}:{lineno} imports '{mod_name}'"
                    )

            if layer in FORBIDDEN_INTERNAL_IMPORTS:
                for forbidden_layer in sorted(FORBIDDEN_INTERNAL_IMPORTS[layer]):
                    prefix = f"{PACKAGE_NAME}.{forbidden_layer}"
                    if mod_name == prefix or mod_name.startswith(f"{prefix}."):
                        violations.append(
                            f"[LAYER FORBIDDEN] {py_file}:{lineno} in '{layer}' imports "
                            f"forbidden '{mod_name}'"
                        )

    return violations


def main() -> None:
    src_path = Path(__file__).resolve().parent.parent
    violations = check_boundaries(src_path)

    if violations:
        print(
            f"FAILED: Found {len(violations)} architectural boundary violation(s):", file=sys.stderr
        )
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        sys.exit(1)
    print("OK: All architectural boundary checks passed successfully.")


__all__ = [
    "FORBIDDEN_EXTERNAL_PACKAGES",
    "FORBIDDEN_INTERNAL_IMPORTS",
    "PACKAGE_NAME",
    "check_boundaries",
    "get_imported_modules",
    "layer_of",
    "module_name_for",
    "resolve_import_target",
]
