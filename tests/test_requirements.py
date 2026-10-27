"""requirements.txt must list exactly the third-party libraries the code uses: nothing missing, nothing unused."""
import ast
import json
import re
import sys

from src import config

IMPORT_TO_PACKAGE = {"sklearn": "scikit-learn"}
TOOLING_ONLY = {"ipykernel", "nbconvert"}            # needed to run the notebook, never imported


def _imports_in(source: str) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _all_imports() -> set[str]:
    sources = [p.read_text(encoding="utf-8") for folder in ("src", "tests") for p in (config.PROJECT_ROOT / folder).glob("*.py")]
    notebook = json.loads((config.PROJECT_ROOT / "notebooks" / "churn_analysis.ipynb").read_text(encoding="utf-8"))
    sources += ["\n".join(l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("%", "!"))) for c in notebook["cells"] if c["cell_type"] == "code"]
    names = set().union(*(_imports_in(s) for s in sources))
    return {n for n in names if n not in sys.stdlib_module_names and n != "src" and n != "tests"}


def _requirements() -> set[str]:
    lines = (config.PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return {re.split(r"[~=<>! ]", l.strip())[0].lower() for l in lines if l.strip() and not l.lstrip().startswith("#")}


def test_every_third_party_import_is_listed():
    needed = {IMPORT_TO_PACKAGE.get(n, n).lower() for n in _all_imports()}
    assert needed <= _requirements(), f"imported but not in requirements.txt: {sorted(needed - _requirements())}"


def test_nothing_unused_is_listed():
    needed = {IMPORT_TO_PACKAGE.get(n, n).lower() for n in _all_imports()}
    assert _requirements() - needed <= TOOLING_ONLY, f"listed but never imported: {sorted(_requirements() - needed - TOOLING_ONLY)}"


def test_the_pins_match_the_versions_installed_here():
    import importlib.metadata as md
    from packaging.requirements import Requirement
    from packaging.version import Version

    lines = (
        config.PROJECT_ROOT / "requirements.txt"
    ).read_text(encoding="utf-8").splitlines()

    for line in lines:
        # Remove comments and ignore blank lines.
        line = line.split("#", 1)[0].strip()
        if not line:
            continue

        requirement = Requirement(line)
        installed = Version(md.version(requirement.name))

        assert installed in requirement.specifier, (
            f"{requirement.name}: installed version {installed} "
            f"does not satisfy {requirement.specifier}"
        )
