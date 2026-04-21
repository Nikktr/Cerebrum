"""Property-based tests for ResultsWriter.compute_summary_statistics using Hypothesis.

Feature: kernel-managed-shared-memory, Property 3: Summary statistics match arithmetic definitions
"""

import math
import statistics
import sys
sys.path.insert(0, ".")

from hypothesis import given, settings
from hypothesis.strategies import (
    integers,
    floats,
    just,
    lists,
    composite,
)

from benchmarks.shared_memory.results import ResultsWriter
from benchmarks.shared_memory.models import TrialResult, MemoryCounts


@composite
def trial_result_strategy(draw):
    """Generate a non-failed TrialResult with random scores, latency, and memory counts."""
    return TrialResult(
        trial_index=draw(integers(min_value=0, max_value=100)),
        condition=draw(just("phase1")),
        profile_usage_score=draw(integers(min_value=1, max_value=5)),
        task_usage_score=draw(integers(min_value=1, max_value=5)),
        integration_score=draw(integers(min_value=1, max_value=5)),
        latency_seconds=draw(floats(min_value=0.1, max_value=30.0)),
        memory_counts=MemoryCounts(
            total=draw(integers(min_value=0, max_value=100)),
            shared=draw(integers(min_value=0, max_value=100)),
            private=draw(integers(min_value=0, max_value=100)),
        ),
        failed=draw(just(False)),
    )


trial_result_lists = lists(trial_result_strategy(), min_size=2, max_size=20)


# Feature: kernel-managed-shared-memory, Property 3: Summary statistics match arithmetic definitions
class TestSummaryStatisticsMatchArithmetic:
    """**Validates: Requirements 11.1, 11.2**"""

    @given(trials=trial_result_lists)
    @settings(max_examples=100)
    def test_summary_mean_and_std_match_statistics_module(self, trials):
        """For any non-empty list of non-failed TrialResult objects, the
        ConditionSummary computed by compute_summary_statistics produces mean
        values equal to statistics.mean and std values equal to statistics.stdev
        for all metric fields."""
        writer = ResultsWriter(output_dir="/tmp/test_output")
        summary = writer.compute_summary_statistics(trials)

        # Extract raw values from trials (mirroring what compute_summary_statistics does)
        profile_vals = [float(t.profile_usage_score) for t in trials]
        task_vals = [float(t.task_usage_score) for t in trials]
        integration_vals = [float(t.integration_score) for t in trials]
        latency_vals = [t.latency_seconds for t in trials]
        mem_total_vals = [float(t.memory_counts.total) for t in trials]
        mem_shared_vals = [float(t.memory_counts.shared) for t in trials]
        mem_private_vals = [float(t.memory_counts.private) for t in trials]

        metrics = [
            ("profile_usage", profile_vals, summary.profile_usage),
            ("task_usage", task_vals, summary.task_usage),
            ("integration", integration_vals, summary.integration),
            ("latency", latency_vals, summary.latency),
            ("memory_total", mem_total_vals, summary.memory_total),
            ("memory_shared", mem_shared_vals, summary.memory_shared),
            ("memory_private", mem_private_vals, summary.memory_private),
        ]

        for name, vals, stats in metrics:
            expected_mean = statistics.mean(vals)
            expected_std = statistics.stdev(vals) if len(vals) > 1 else 0.0

            assert math.isclose(stats.mean, expected_mean, rel_tol=1e-9), (
                f"{name}: mean {stats.mean} != expected {expected_mean}"
            )
            assert math.isclose(stats.std, expected_std, rel_tol=1e-9), (
                f"{name}: std {stats.std} != expected {expected_std}"
            )

    @given(trials=lists(trial_result_strategy(), min_size=1, max_size=1))
    @settings(max_examples=100)
    def test_single_element_std_is_zero(self, trials):
        """For a single-element list, std should be 0.0 for all metrics."""
        writer = ResultsWriter(output_dir="/tmp/test_output")
        summary = writer.compute_summary_statistics(trials)

        t = trials[0]
        expected = {
            "profile_usage": float(t.profile_usage_score),
            "task_usage": float(t.task_usage_score),
            "integration": float(t.integration_score),
            "latency": t.latency_seconds,
            "memory_total": float(t.memory_counts.total),
            "memory_shared": float(t.memory_counts.shared),
            "memory_private": float(t.memory_counts.private),
        }

        for name, stats in [
            ("profile_usage", summary.profile_usage),
            ("task_usage", summary.task_usage),
            ("integration", summary.integration),
            ("latency", summary.latency),
            ("memory_total", summary.memory_total),
            ("memory_shared", summary.memory_shared),
            ("memory_private", summary.memory_private),
        ]:
            assert stats.std == 0.0, (
                f"{name}: std should be 0.0 for single element, got {stats.std}"
            )
            assert math.isclose(stats.mean, expected[name], rel_tol=1e-9), (
                f"{name}: mean {stats.mean} != expected {expected[name]}"
            )


if __name__ == "__main__":
    test = TestSummaryStatisticsMatchArithmetic()
    print("Running Property 3: Summary statistics match arithmetic definitions...")
    test.test_summary_mean_and_std_match_statistics_module()
    print("PASSED: Property 3 (mean and std for lists of 2-20)")
    test.test_single_element_std_is_zero()
    print("PASSED: Property 3 (single element std is 0.0)")
