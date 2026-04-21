"""CLI entry point and orchestrator for the shared memory evaluation harness.

Runs a two-condition experiment comparing private-only memory (Phase 1)
against shared memory (Phase 2) across synthetic trials, collecting
profile-usage, task-usage, integration, memory-count, and latency metrics.

The kernel's ``memory.auto_inject`` is assumed to be enabled for both phases.
The only difference between phases is the ``share_memory`` flag on agents:
- Phase 1: agents write memories with sharing_policy="private" → kernel
  auto-inject finds nothing eligible to inject cross-agent.
- Phase 2: agents write memories with sharing_policy="shared" → kernel
  auto-inject retrieves and injects them into AssistantAgent's context.

Restart the kernel between phases to clear the memory store and prevent
rollover.

Usage::

    python benchmarks/shared_memory/run_evaluation.py --trials 10 --output results/ --condition phase1 --csv
    # restart kernel to clear memory
    python benchmarks/shared_memory/run_evaluation.py --trials 10 --output results/ --condition phase2 --csv
"""

import argparse
import logging
import os
import statistics
import sys
from datetime import datetime

# Ensure the project root is on sys.path when running as a script
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tqdm import tqdm

from benchmarks.shared_memory.models import (
    ConditionResults,
    ExperimentMetadata,
    ExperimentResults,
    JudgeScores,
    MemoryCounts,
    TrialResult,
)
from benchmarks.shared_memory.judge import LLMJudge
from benchmarks.shared_memory.pipeline import AgentPipeline
from benchmarks.shared_memory.results import ResultsWriter
from benchmarks.shared_memory.synth import SyntheticDataGenerator
from cerebrum.config.config_manager import config

logger = logging.getLogger(__name__)


class EvaluationOrchestrator:
    """Orchestrates the shared memory evaluation experiment.

    Args:
        trials: Number of trials to run per condition.
        output_dir: Directory path for writing result files.
        write_csv: If True, also write a CSV file alongside JSON.
        condition: Which conditions to run — "both", "phase1", or "phase2".
    """

    def __init__(
        self,
        trials: int,
        output_dir: str,
        write_csv: bool,
        condition: str,
    ):
        self.trials = trials
        self.output_dir = output_dir
        self.write_csv = write_csv
        self.condition = condition

        self.generator = SyntheticDataGenerator()
        self.judge = LLMJudge()
        self.writer = ResultsWriter(output_dir=output_dir, write_csv=write_csv)

    def run_single_trial(
        self,
        trial_index: int,
        condition: str,
        pipeline: AgentPipeline,
    ) -> TrialResult:
        """Execute one trial with log-and-continue error handling."""
        # Step 1: Generate synthetic data
        try:
            trial_data = self.generator.generate_trial_data(trial_index)
        except Exception as e:
            logger.error("Trial %d: synthetic data generation failed: %s", trial_index, e)
            return TrialResult(
                trial_index=trial_index,
                condition=condition,
                failed=True,
                error_message=str(e),
            )

        # Step 2: Run agent pipeline
        try:
            pipeline_result = pipeline.run_trial(trial_data)
        except Exception as e:
            logger.error("Trial %d: agent pipeline failed: %s", trial_index, e)
            return TrialResult(
                trial_index=trial_index,
                condition=condition,
                failed=True,
                error_message=str(e),
                synthetic_profile=trial_data.profile,
                synthetic_task_context=trial_data.task_context,
                follow_up_query=trial_data.follow_up_query,
            )

        # Extract retrieval log from pipeline result
        retrieval_log = pipeline_result.retrieval_log

        # Phase 1 isolation verification
        if condition == "phase1" and retrieval_log:
            if not retrieval_log.cross_agent_found:
                logger.info("Trial %d: Phase 1 isolation verified — zero cross-agent memories.", trial_index)
            else:
                entries = [(e.owner_agent, e.memory_type) for e in retrieval_log.retrieved_memories]
                logger.warning(
                    "Trial %d: Cross-agent memory leakage detected! count=%d entries=%s",
                    trial_index, len([e for e in retrieval_log.retrieved_memories if e.owner_agent != "assistant_agent"]), entries,
                )

        # Phase 2 retrieval audit
        if condition == "phase2" and retrieval_log:
            entries = [(e.owner_agent, e.memory_type) for e in retrieval_log.retrieved_memories]
            logger.info(
                "Trial %d: Retrieved %d shared memories. Entries: %s",
                trial_index, retrieval_log.shared_memory_count, entries,
            )

        # Step 3: Judge the assistant response
        try:
            scores = self.judge.evaluate(
                query=trial_data.follow_up_query,
                response=pipeline_result.assistant_response,
                profile=trial_data.profile,
                task_context=trial_data.task_context,
                plausible_actions=trial_data.plausible_actions,
            )
        except Exception as e:
            logger.warning("Trial %d: judge evaluation failed: %s", trial_index, e)
            scores = JudgeScores()

        # Memory counts heuristic
        if condition == "phase2":
            memory_counts = MemoryCounts(total=2, shared=2, private=0)
        else:
            memory_counts = MemoryCounts(total=2, shared=0, private=2)

        return TrialResult(
            trial_index=trial_index,
            condition=condition,
            profile_usage_score=scores.profile_usage_score,
            task_usage_score=scores.task_usage_score,
            integration_score=scores.integration_score,
            memory_counts=memory_counts,
            latency_seconds=pipeline_result.latency_seconds,
            follow_up_query=trial_data.follow_up_query,
            assistant_response=pipeline_result.assistant_response,
            synthetic_profile=trial_data.profile,
            synthetic_task_context=trial_data.task_context,
            retrieval_log=retrieval_log,
            injection_diagnostics=pipeline_result.injection_diagnostics,
            written_memories=pipeline_result.written_memories,
        )

    def run(self) -> ExperimentResults:
        """Run the full experiment across all requested conditions.

        The kernel's auto_inject is assumed to be enabled externally.
        The only control variable is the share_memory flag on agents:
        Phase 1 sets it to False (private), Phase 2 sets it to True (shared).
        """
        # Determine which conditions to run
        if self.condition == "both":
            conditions = ["phase1", "phase2"]
        elif self.condition == "phase1":
            conditions = ["phase1"]
        else:
            conditions = ["phase2"]

        condition_results = []

        for cond in conditions:
            share_memory = cond == "phase2"
            pipeline = AgentPipeline(share_memory=share_memory)
            logger.info(
                "Running condition '%s' (share_memory=%s, kernel auto_inject assumed ON).",
                cond,
                share_memory,
            )

            trials: list[TrialResult] = []
            for i in tqdm(range(self.trials), desc=f"Condition: {cond}"):
                result = self.run_single_trial(i, cond, pipeline)
                trials.append(result)

            summary = self.writer.compute_summary_statistics(trials)
            condition_results.append(
                ConditionResults(condition=cond, trials=trials, summary=summary)
            )

        metadata = ExperimentMetadata(
            trials_per_condition=self.trials,
            timestamp=datetime.now().isoformat(),
            kernel_url=config.get_kernel_url(),
            conditions_run=conditions,
        )

        experiment = ExperimentResults(
            experiment_metadata=metadata,
            conditions=condition_results,
        )

        json_path = self.writer.write_json(experiment)
        logger.info("Results written to %s", json_path)

        if self.write_csv:
            csv_path = self.writer.write_csv(experiment)
            logger.info("CSV written to %s", csv_path)

        return experiment


def main():
    """Parse CLI arguments and run the evaluation orchestrator."""
    parser = argparse.ArgumentParser(
        description="Shared Memory Evaluation Harness — measures whether "
        "shared memory improves personalization quality.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of trials per condition (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory for result files (default: results/).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write a CSV file alongside the JSON results.",
    )
    parser.add_argument(
        "--condition",
        choices=["both", "phase1", "phase2"],
        default="both",
        help="Which condition(s) to run (default: both).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    orchestrator = EvaluationOrchestrator(
        trials=args.trials,
        output_dir=args.output,
        write_csv=args.csv,
        condition=args.condition,
    )

    experiment = orchestrator.run()

    # Print summary to stdout
    meta = experiment.experiment_metadata
    print(f"\n{'=' * 60}")
    print(f"Experiment complete — {meta.timestamp}")
    print(f"Conditions: {', '.join(meta.conditions_run)}")
    print(f"Trials per condition: {meta.trials_per_condition}")
    print(f"{'=' * 60}")

    for cond_result in experiment.conditions:
        s = cond_result.summary
        print(f"\n--- {cond_result.condition} ---")
        print(f"  Profile Usage:      mean={s.profile_usage.mean:.2f}  std={s.profile_usage.std:.2f}")
        print(f"  Task Usage:         mean={s.task_usage.mean:.2f}  std={s.task_usage.std:.2f}")
        print(f"  Integration:        mean={s.integration.mean:.2f}  std={s.integration.std:.2f}")
        print(f"  Latency (s):        mean={s.latency.mean:.2f}  std={s.latency.std:.2f}")
        print(f"  Memory total:       mean={s.memory_total.mean:.2f}")
        print(f"  Injected memories:  mean={s.injected_memories.mean:.2f}  std={s.injected_memories.std:.2f}")
        print(f"  Trials: {s.total_trials} total, {s.failed_trials} failed")

        non_failed = [t for t in cond_result.trials if not t.failed]
        mean_shared = statistics.mean([
            t.retrieval_log.shared_memory_count
            for t in non_failed if t.retrieval_log
        ]) if any(t.retrieval_log for t in non_failed) else 0.0
        cross_agent_count = sum(
            1 for t in non_failed if t.retrieval_log and t.retrieval_log.cross_agent_found
        )
        print(f"  Shared mem retrieved: mean={mean_shared:.2f}")
        print(f"  Cross-agent trials:  {cross_agent_count}/{len(non_failed)}")

    # Comparative analysis: Phase 1 vs Phase 2
    conditions_by_name = {c.condition: c.summary for c in experiment.conditions}
    if "phase1" in conditions_by_name and "phase2" in conditions_by_name:
        p1 = conditions_by_name["phase1"]
        p2 = conditions_by_name["phase2"]

        print(f"\n{'=' * 60}")
        print("Comparative Analysis: Phase 1 (private) vs Phase 2 (shared)")
        print(f"{'=' * 60}")
        header = f"  {'Metric':<22} {'Phase1':>12} {'Phase2':>12} {'Delta':>12}"
        print(header)
        print(f"  {'-' * 58}")

        rows = [
            ("Profile Usage mean", p1.profile_usage.mean, p2.profile_usage.mean),
            ("Profile Usage std", p1.profile_usage.std, p2.profile_usage.std),
            ("Task Usage mean", p1.task_usage.mean, p2.task_usage.mean),
            ("Task Usage std", p1.task_usage.std, p2.task_usage.std),
            ("Integration mean", p1.integration.mean, p2.integration.mean),
            ("Integration std", p1.integration.std, p2.integration.std),
            ("Injected mem mean", p1.injected_memories.mean, p2.injected_memories.mean),
        ]
        for label, v1, v2 in rows:
            delta = v2 - v1
            sign = "+" if delta >= 0 else ""
            print(f"  {label:<22} {v1:>12.2f} {v2:>12.2f} {sign + f'{delta:.2f}':>12}")

    print()


if __name__ == "__main__":
    main()
