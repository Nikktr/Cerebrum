"""Unit tests for EvaluationOrchestrator — simplified config model.

The orchestrator no longer toggles kernel auto_inject. It assumes
auto_inject is always on in the kernel. The only control variable is
the share_memory flag on agents (Phase 1 = False, Phase 2 = True).

Validates: Requirements 7.1, 7.2, 8.1, 8.3
"""

import sys
sys.path.insert(0, ".")

from unittest.mock import patch, MagicMock

from benchmarks.shared_memory.models import (
    ConditionSummary,
    MemoryCounts,
    SummaryStatistics,
    TrialResult,
)


def _make_trial_result(trial_index, condition):
    """Return a minimal non-failed TrialResult for mocking."""
    return TrialResult(
        trial_index=trial_index,
        condition=condition,
        profile_usage_score=3,
        task_usage_score=3,
        integration_score=3,
        memory_counts=MemoryCounts(total=2, shared=0, private=2),
        latency_seconds=1.0,
    )


def _make_zero_stats():
    """Return a SummaryStatistics with all zeros."""
    return SummaryStatistics(mean=0.0, std=0.0, min=0.0, max=0.0)


def _make_condition_summary():
    """Return a minimal ConditionSummary for mocking."""
    s = _make_zero_stats()
    return ConditionSummary(
        profile_usage=s,
        task_usage=s,
        integration=s,
        latency=s,
        memory_total=s,
        memory_shared=s,
        memory_private=s,
        injected_memories=s,
        total_trials=1,
        failed_trials=0,
    )


def _build_orchestrator(condition="both"):
    """Instantiate an EvaluationOrchestrator with patched dependencies."""
    from benchmarks.shared_memory.run_evaluation import EvaluationOrchestrator

    with patch(
        "benchmarks.shared_memory.run_evaluation.SyntheticDataGenerator"
    ), patch(
        "benchmarks.shared_memory.run_evaluation.LLMJudge"
    ), patch(
        "benchmarks.shared_memory.run_evaluation.ResultsWriter"
    ) as mock_writer_cls:
        mock_writer_instance = mock_writer_cls.return_value
        mock_writer_instance.compute_summary_statistics.return_value = _make_condition_summary()
        mock_writer_instance.write_json.return_value = "/tmp/results.json"
        mock_writer_instance.write_csv.return_value = "/tmp/results.csv"

        orch = EvaluationOrchestrator(
            trials=1,
            output_dir="/tmp/test_output",
            write_csv=False,
            condition=condition,
        )
    return orch


def test_phase1_uses_share_memory_false():
    """Phase 1 creates pipeline with share_memory=False. (Req 7.1)"""
    orch = _build_orchestrator(condition="phase1")

    pipelines_created = []
    original_init = None

    from benchmarks.shared_memory.pipeline import AgentPipeline
    original_init = AgentPipeline.__init__

    def tracking_init(self, share_memory):
        pipelines_created.append(share_memory)
        original_init(self, share_memory)

    with patch(
        "benchmarks.shared_memory.run_evaluation.config.get_kernel_url",
        return_value="http://localhost:8000",
    ), patch.object(
        orch, "run_single_trial",
        return_value=_make_trial_result(0, "phase1"),
    ), patch(
        "benchmarks.shared_memory.run_evaluation.tqdm",
        side_effect=lambda iterable, **kw: iterable,
    ), patch.object(
        AgentPipeline, "__init__", tracking_init,
    ):
        orch.run()

    assert len(pipelines_created) == 1
    assert pipelines_created[0] is False, (
        f"Phase 1 should use share_memory=False, got {pipelines_created[0]}"
    )
    print("PASSED: Phase 1 uses share_memory=False")


def test_phase2_uses_share_memory_true():
    """Phase 2 creates pipeline with share_memory=True. (Req 8.1)"""
    orch = _build_orchestrator(condition="phase2")

    pipelines_created = []

    from benchmarks.shared_memory.pipeline import AgentPipeline
    original_init = AgentPipeline.__init__

    def tracking_init(self, share_memory):
        pipelines_created.append(share_memory)
        original_init(self, share_memory)

    with patch(
        "benchmarks.shared_memory.run_evaluation.config.get_kernel_url",
        return_value="http://localhost:8000",
    ), patch.object(
        orch, "run_single_trial",
        return_value=_make_trial_result(0, "phase2"),
    ), patch(
        "benchmarks.shared_memory.run_evaluation.tqdm",
        side_effect=lambda iterable, **kw: iterable,
    ), patch.object(
        AgentPipeline, "__init__", tracking_init,
    ):
        orch.run()

    assert len(pipelines_created) == 1
    assert pipelines_created[0] is True, (
        f"Phase 2 should use share_memory=True, got {pipelines_created[0]}"
    )
    print("PASSED: Phase 2 uses share_memory=True")


def test_no_config_update_called():
    """Orchestrator does not call config.update (auto_inject managed by kernel)."""
    orch = _build_orchestrator(condition="both")

    with patch(
        "benchmarks.shared_memory.run_evaluation.config.get_kernel_url",
        return_value="http://localhost:8000",
    ), patch(
        "benchmarks.shared_memory.run_evaluation.config.update",
    ) as mock_update, patch.object(
        orch, "run_single_trial",
        return_value=_make_trial_result(0, "phase1"),
    ), patch(
        "benchmarks.shared_memory.run_evaluation.tqdm",
        side_effect=lambda iterable, **kw: iterable,
    ):
        orch.run()

    mock_update.assert_not_called()
    print("PASSED: config.update is not called (auto_inject managed by kernel)")


if __name__ == "__main__":
    test_phase1_uses_share_memory_false()
    test_phase2_uses_share_memory_true()
    test_no_config_update_called()
    print("\nAll tests passed.")
