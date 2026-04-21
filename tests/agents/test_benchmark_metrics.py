"""Unit tests for benchmark metric recording and comparative output.

Validates: Requirements 9.1, 9.2, 9.3, 9.7, 9.8, 11.4

Tests verify that TrialResult contains all expected metric fields and
that the comparative analysis section is printed to stdout when both
Phase 1 and Phase 2 results are present.
"""

import sys
sys.path.insert(0, ".")

import contextlib
import io
from unittest.mock import patch, MagicMock

from benchmarks.shared_memory.models import (
    ConditionResults,
    ConditionSummary,
    ExperimentMetadata,
    ExperimentResults,
    InjectedMemoryEntry,
    InjectionDiagnostics,
    MemoryCounts,
    SummaryStatistics,
    TrialResult,
    WrittenMemoryRecord,
)


def _make_summary(profile_mean=3.0, task_mean=3.0, integration_mean=3.0,
                  injected_mean=0.0):
    """Return a ConditionSummary with configurable means."""
    def _stats(mean):
        return SummaryStatistics(mean=mean, std=0.5, min=1.0, max=5.0)

    return ConditionSummary(
        profile_usage=_stats(profile_mean),
        task_usage=_stats(task_mean),
        integration=_stats(integration_mean),
        latency=_stats(1.5),
        memory_total=_stats(2.0),
        memory_shared=_stats(1.0),
        memory_private=_stats(1.0),
        injected_memories=_stats(injected_mean),
        total_trials=3,
        failed_trials=0,
    )


def test_trial_result_has_all_fields():
    """TrialResult contains scores, injection_diagnostics, written_memories,
    and latency_seconds with correct values. (Req 9.1, 9.2, 9.3, 9.7)"""

    diagnostics = InjectionDiagnostics(
        injected_count=2,
        injected_memories=[
            InjectedMemoryEntry(
                owner_agent="profile_agent",
                memory_type="profile",
                match_score=0.95,
            ),
            InjectedMemoryEntry(
                owner_agent="task_agent",
                memory_type="task_context",
                match_score=0.88,
            ),
        ],
    )

    written = [
        WrittenMemoryRecord(
            agent_name="profile_agent",
            memory_type="profile",
            sharing_policy="shared",
            user_id="user_42",
        ),
        WrittenMemoryRecord(
            agent_name="task_agent",
            memory_type="task_context",
            sharing_policy="shared",
            user_id="user_42",
        ),
    ]

    trial = TrialResult(
        trial_index=0,
        condition="phase2",
        profile_usage_score=4,
        task_usage_score=5,
        integration_score=3,
        memory_counts=MemoryCounts(total=2, shared=2, private=0),
        latency_seconds=1.23,
        injection_diagnostics=diagnostics,
        written_memories=written,
    )

    # Scores (Req 9.1, 9.2, 9.3)
    assert trial.profile_usage_score == 4, (
        f"Expected profile_usage_score=4, got {trial.profile_usage_score}"
    )
    assert trial.task_usage_score == 5, (
        f"Expected task_usage_score=5, got {trial.task_usage_score}"
    )
    assert trial.integration_score == 3, (
        f"Expected integration_score=3, got {trial.integration_score}"
    )

    # Latency (Req 9.7)
    assert trial.latency_seconds == 1.23, (
        f"Expected latency_seconds=1.23, got {trial.latency_seconds}"
    )

    # Injection diagnostics (Req 9.4, 9.5)
    assert trial.injection_diagnostics is not None
    assert trial.injection_diagnostics.injected_count == 2
    assert len(trial.injection_diagnostics.injected_memories) == 2
    assert trial.injection_diagnostics.injected_memories[0].owner_agent == "profile_agent"
    assert trial.injection_diagnostics.injected_memories[1].memory_type == "task_context"

    # Written memories (Req 9.6)
    assert len(trial.written_memories) == 2
    assert trial.written_memories[0].agent_name == "profile_agent"
    assert trial.written_memories[0].sharing_policy == "shared"
    assert trial.written_memories[1].memory_type == "task_context"
    assert trial.written_memories[1].user_id == "user_42"

    print("PASSED: TrialResult contains all expected metric fields")


def test_comparative_output_printed():
    """Comparative analysis is printed to stdout when both phases are present.
    (Req 11.4)"""

    p1_summary = _make_summary(
        profile_mean=2.0, task_mean=2.0, integration_mean=2.0,
        injected_mean=0.0,
    )
    p2_summary = _make_summary(
        profile_mean=4.0, task_mean=4.0, integration_mean=4.0,
        injected_mean=2.0,
    )

    # Build a minimal TrialResult for each condition so the per-condition
    # print loop doesn't crash on retrieval_log access.
    p1_trial = TrialResult(
        trial_index=0, condition="phase1",
        profile_usage_score=2, task_usage_score=2, integration_score=2,
        latency_seconds=1.0,
    )
    p2_trial = TrialResult(
        trial_index=0, condition="phase2",
        profile_usage_score=4, task_usage_score=4, integration_score=4,
        latency_seconds=1.0,
    )

    experiment = ExperimentResults(
        experiment_metadata=ExperimentMetadata(
            trials_per_condition=1,
            timestamp="2025-01-01T00:00:00",
            kernel_url="http://localhost:8000",
            conditions_run=["phase1", "phase2"],
        ),
        conditions=[
            ConditionResults(
                condition="phase1", trials=[p1_trial], summary=p1_summary,
            ),
            ConditionResults(
                condition="phase2", trials=[p2_trial], summary=p2_summary,
            ),
        ],
    )

    # Patch main() dependencies so we can capture the stdout printing logic.
    # The simplest approach: call main() with mocked orchestrator that returns
    # our pre-built experiment, and capture stdout.
    from benchmarks.shared_memory.run_evaluation import main

    mock_orch_instance = MagicMock()
    mock_orch_instance.run.return_value = experiment

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), \
         patch("benchmarks.shared_memory.run_evaluation.argparse.ArgumentParser") as mock_parser_cls, \
         patch("benchmarks.shared_memory.run_evaluation.EvaluationOrchestrator", return_value=mock_orch_instance), \
         patch("benchmarks.shared_memory.run_evaluation.logging"):

        # Configure the mock argument parser
        mock_args = MagicMock()
        mock_args.trials = 1
        mock_args.output = "/tmp/test"
        mock_args.csv = False
        mock_args.condition = "both"
        mock_parser_cls.return_value.parse_args.return_value = mock_args

        main()

    output = captured.getvalue()

    # Verify comparative analysis section is present
    assert "Comparative Analysis" in output, (
        f"Expected 'Comparative Analysis' in stdout, got:\n{output}"
    )

    # Verify expected metric labels appear
    assert "Profile Usage" in output, (
        f"Expected 'Profile Usage' in stdout, got:\n{output}"
    )
    assert "Task Usage" in output, (
        f"Expected 'Task Usage' in stdout, got:\n{output}"
    )
    assert "Integration" in output, (
        f"Expected 'Integration' in stdout, got:\n{output}"
    )
    assert "Injected mem" in output, (
        f"Expected 'Injected mem' in stdout, got:\n{output}"
    )

    # Verify phase headers are present
    assert "Phase1" in output or "phase1" in output, (
        f"Expected phase1 reference in stdout, got:\n{output}"
    )
    assert "Phase2" in output or "phase2" in output, (
        f"Expected phase2 reference in stdout, got:\n{output}"
    )

    # Verify delta column is present (shows difference between phases)
    assert "Delta" in output, (
        f"Expected 'Delta' column in stdout, got:\n{output}"
    )

    print("PASSED: Comparative analysis is printed to stdout")


if __name__ == "__main__":
    test_trial_result_has_all_fields()
    test_comparative_output_printed()
    print("\nAll tests passed.")
