"""Dependency single-source-of-truth guard.

pyproject.toml is authoritative; requirements.txt (consumed by the Docker
image) and requirements-dev.txt (plain-pip bootstrap) are derived mirrors.
These tests fail the suite if the mirrors drift.
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _requirement_lines(name):
    return sorted(
        line.strip()
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(("#", "-r"))
    )


def test_runtime_dependencies_mirror_requirements_txt():
    assert sorted(_pyproject()["project"]["dependencies"]) == _requirement_lines(
        "requirements.txt"
    )


def test_dev_dependencies_mirror_requirements_dev_txt():
    """The bootstrap file is the hatch env dependencies plus hatch itself,
    pinned: hatch is the outer tool and cannot be a dependency of the
    environment it manages."""
    hatch_env = _pyproject()["tool"]["hatch"]["envs"]["default"]
    dev_lines = _requirement_lines("requirements-dev.txt")
    tooling = [line for line in dev_lines if line.startswith("hatch==")]
    assert len(tooling) == 1
    assert sorted(hatch_env["dependencies"]) == sorted(
        set(dev_lines) - set(tooling)
    )


def test_dev_requirements_include_runtime_requirements():
    text = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "-r requirements.txt" in text


def test_hatch_scripts_expose_exactly_the_clean_command_surface():
    """Generic plumbing plus the one focused benchmark; no per-experiment
    commands for modules that no longer exist."""
    scripts = _pyproject()["tool"]["hatch"]["envs"]["default"]["scripts"]
    assert set(scripts) == {"test", "test-nodb", "verify", "serve", "benchmark"}
    assert scripts["benchmark"] == "python -m app.benchmark {args}"
