"""Tests for the GitHub Actions workflow configurations.

Validates that ci.yml and retrain.yml exist and have the correct
structure: triggers, Python version, steps, etc.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CI_WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"


@pytest.fixture
def workflow() -> dict:
    """Load and parse the CI workflow YAML."""
    assert CI_WORKFLOW_PATH.exists(), f"CI workflow not found at {CI_WORKFLOW_PATH}"
    text = CI_WORKFLOW_PATH.read_text()
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict), "Workflow YAML must be a mapping"
    return parsed


def _get_triggers(workflow: dict) -> dict:
    """Return the trigger block — PyYAML parses bare ``on`` as ``True``."""
    if "on" in workflow:
        return workflow["on"]
    if True in workflow:
        return workflow[True]
    raise KeyError("Workflow missing 'on' / True trigger block")


class TestCIWorkflowTriggers:
    """Workflow must trigger on push-to-main and on pull requests."""

    def test_triggers_on_push_to_main(self, workflow: dict) -> None:
        triggers = _get_triggers(workflow)
        assert "push" in triggers, "Workflow must trigger on push"
        push_branches = triggers["push"].get("branches", [])
        assert "main" in push_branches, "Push trigger must include 'main' branch"

    def test_triggers_on_pull_request(self, workflow: dict) -> None:
        triggers = _get_triggers(workflow)
        assert "pull_request" in triggers, "Workflow must trigger on pull_request"


class TestCIWorkflowJob:
    """Workflow must define a test job with the right steps."""

    def test_has_test_job(self, workflow: dict) -> None:
        assert "jobs" in workflow, "Workflow missing 'jobs' block"
        assert "test" in workflow["jobs"], "Workflow must have a 'test' job"

    def test_runs_on_ubuntu(self, workflow: dict) -> None:
        runs_on = workflow["jobs"]["test"].get("runs-on", "")
        assert "ubuntu" in runs_on, "Test job must run on ubuntu"

    def test_uses_python_310(self, workflow: dict) -> None:
        steps = workflow["jobs"]["test"]["steps"]
        setup_step = _find_step_by_uses(steps, "setup-python")
        assert setup_step is not None, "Must have a setup-python step"
        version = str(setup_step.get("with", {}).get("python-version", ""))
        assert "3.10" in version, "Must use Python 3.10"

    def test_installs_dependencies(self, workflow: dict) -> None:
        run_text = _collect_run_text(workflow["jobs"]["test"]["steps"])
        assert "requirements.txt" in run_text, "Must install from requirements.txt"

    def test_runs_ruff_lint(self, workflow: dict) -> None:
        run_text = _collect_run_text(workflow["jobs"]["test"]["steps"])
        assert "ruff" in run_text, "Must run ruff linting"

    def test_runs_pytest_with_coverage(self, workflow: dict) -> None:
        run_text = _collect_run_text(workflow["jobs"]["test"]["steps"])
        assert "pytest" in run_text, "Must run pytest"
        assert "--cov" in run_text, "Must run pytest with --cov flag"


def _find_step_by_uses(steps: list[dict], keyword: str) -> dict | None:
    for step in steps:
        if keyword in step.get("uses", ""):
            return step
    return None


def _collect_run_text(steps: list[dict]) -> str:
    return " ".join(step.get("run", "") for step in steps)


# ---------------------------------------------------------------------------
# Retrain workflow
# ---------------------------------------------------------------------------

RETRAIN_WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "retrain.yml"


@pytest.fixture
def retrain_workflow() -> dict:
    """Load and parse the retrain workflow YAML."""
    assert RETRAIN_WORKFLOW_PATH.exists(), f"Retrain workflow not found at {RETRAIN_WORKFLOW_PATH}"
    text = RETRAIN_WORKFLOW_PATH.read_text()
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict)
    return parsed


class TestRetrainWorkflowTriggers:
    def test_has_schedule(self, retrain_workflow: dict) -> None:
        triggers = _get_triggers(retrain_workflow)
        assert "schedule" in triggers, "Retrain must have a schedule trigger"

    def test_has_workflow_dispatch(self, retrain_workflow: dict) -> None:
        triggers = _get_triggers(retrain_workflow)
        assert "workflow_dispatch" in triggers, "Retrain must support manual trigger"


class TestRetrainWorkflowJob:
    def test_has_retrain_job(self, retrain_workflow: dict) -> None:
        assert "retrain" in retrain_workflow["jobs"]

    def test_runs_dvc_repro(self, retrain_workflow: dict) -> None:
        run_text = _collect_run_text(retrain_workflow["jobs"]["retrain"]["steps"])
        assert "dvc repro" in run_text, "Must run dvc repro"

    def test_runs_train(self, retrain_workflow: dict) -> None:
        run_text = _collect_run_text(retrain_workflow["jobs"]["retrain"]["steps"])
        assert "train.py" in run_text, "Must run train.py"

    def test_runs_drift_check(self, retrain_workflow: dict) -> None:
        run_text = _collect_run_text(retrain_workflow["jobs"]["retrain"]["steps"])
        assert "check_drift" in run_text, "Must run drift check"

    def test_commits_with_skip_ci(self, retrain_workflow: dict) -> None:
        run_text = _collect_run_text(retrain_workflow["jobs"]["retrain"]["steps"])
        assert "[skip ci]" in run_text, "Commit must include [skip ci] to prevent loop"
