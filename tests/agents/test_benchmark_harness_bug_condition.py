"""Bug condition exploration tests for the benchmark harness.

These tests are EXPECTED TO FAIL on unfixed code — failure confirms the bugs exist.
DO NOT fix the code to make these pass. The tests encode the expected (correct) behavior.

Bug 1: Judge scoring penalizes concise Phase 2 responses, misapplies generic_penalty,
        and _normalize_judge_keys fails on variant key formats.
Bug 2: RetrievalLog lacks injection_status field, so the pipeline cannot distinguish
        "confirmed no injection" from "injection status unknown."

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
"""

import sys
sys.path.insert(0, ".")

import re
import traceback

from hypothesis import given, settings
from hypothesis import strategies as st

from benchmarks.shared_memory.judge import LLMJudge, _normalize_judge_keys
from benchmarks.shared_memory.models import JudgeScores, RetrievalLog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

results = []


def record(name: str, passed: bool, detail: str = ""):
    """Record a test result."""
    status = "UNEXPECTED PASS" if passed else "EXPECTED FAIL"
    results.append((name, status, detail))
    print(f"  [{status}] {name}")
    if detail:
        print(f"    Detail: {detail}")


# ---------------------------------------------------------------------------
# Bug 1 — Judge Scoring
# ---------------------------------------------------------------------------

def test_a_judge_prompt_lacks_injected_context_awareness():
    """(a) Judge prompt lacks injected-context awareness.

    **Validates: Requirements 1.4**

    The judge prompt should inform the evaluator that the assistant had access
    to injected shared memories. On unfixed code, the prompt has no such
    awareness — this test WILL FAIL (EXPECTED FAIL).
    """
    from benchmarks.shared_memory.models import SyntheticProfile, SyntheticTaskContext

    judge = LLMJudge()
    profile = SyntheticProfile(
        user_name="Alice",
        preferred_tools=["PyTorch", "Jupyter"],
        preferred_language="Python",
        response_style="concise",
    )
    task_context = SyntheticTaskContext(
        current_project="CUDA memory optimizer",
        active_experiment="batch size sweep",
        goals=["reduce GPU memory usage by 30%"],
        blockers=["CUDA OOM on large batches"],
        next_steps=["profile memory with torch.cuda.memory_summary()"],
    )

    query = "What should I try next for my memory issue?"
    response = (
        "Based on your PyTorch project and the CUDA OOM blocker, "
        "I'd recommend using torch.cuda.memory_summary() in your next "
        "experiment to profile the batch size sweep."
    )

    messages = judge._build_judge_prompt(query, response, profile, task_context)
    full_prompt = " ".join(m["content"] for m in messages).lower()

    # The prompt should mention injected shared memories / kernel-injected context
    has_awareness = any(
        phrase in full_prompt
        for phrase in [
            "injected",
            "shared memories",
            "shared memory",
            "kernel-injected",
            "kernel injected",
            "injected context",
            "injected shared",
        ]
    )

    record(
        "a_judge_prompt_injected_context_awareness",
        has_awareness,
        f"Prompt contains injected-context language: {has_awareness}",
    )
    return has_awareness


def test_b_generic_penalty_misapplication():
    """(b) Generic penalty misapplication.

    **Validates: Requirements 1.2**

    When generic_penalty=True and scores are 4/4/4, the penalty cap logic
    (min(score, 2)) unconditionally caps all scores at 2. This test asserts
    the scores are NOT capped — it WILL FAIL on unfixed code because the cap
    is unconditional.
    """
    # Simulate the penalty cap logic from evaluate()
    scores = JudgeScores(
        generic_penalty=True,
        profile_usage_score=4,
        task_usage_score=4,
        integration_score=4,
    )

    # Apply the same penalty logic as in evaluate()
    if scores.generic_penalty is True:
        scores.profile_usage_score = min(scores.profile_usage_score or 2, 2)
        scores.task_usage_score = min(scores.task_usage_score or 2, 2)
        scores.integration_score = min(scores.integration_score or 2, 2)

    # After the penalty, scores should NOT be capped at 2 (expected behavior).
    # On unfixed code, they WILL be capped at 2, so this assertion fails.
    not_capped = (
        scores.profile_usage_score > 2
        or scores.task_usage_score > 2
        or scores.integration_score > 2
    )

    record(
        "b_generic_penalty_misapplication",
        not_capped,
        f"Scores after penalty: profile={scores.profile_usage_score}, "
        f"task={scores.task_usage_score}, integration={scores.integration_score}",
    )
    return not_capped


def test_c_key_normalization_variant_handling():
    """(c) Key normalization variant handling.

    **Validates: Requirements 1.3**

    _normalize_judge_keys should handle variant key formats like title-case,
    CamelCase, and hyphenated keys. On unfixed code, these variants are NOT
    in the key_map, so extraction fails.
    """
    variant_dicts = [
        {"Profile Usage Score": 4, "Task Usage Score": 3, "Integration Score": 5},
        {"profileUsageScore": 4, "taskUsageScore": 3, "integrationScore": 5},
        {"task-usage-score": 3, "profile-usage-score": 4, "integration-score": 5},
        {"PROFILE_USAGE_SCORE": 4, "TASK_USAGE_SCORE": 3, "INTEGRATION_SCORE": 5},
    ]

    all_passed = True
    details = []
    for variant in variant_dicts:
        normalized = _normalize_judge_keys(variant)
        pu = normalized.get("profile_usage_score")
        tu = normalized.get("task_usage_score")
        ig = normalized.get("integration_score")
        ok = pu is not None and tu is not None and ig is not None
        if not ok:
            all_passed = False
            details.append(
                f"Input keys {list(variant.keys())} -> "
                f"profile={pu}, task={tu}, integration={ig}"
            )

    record(
        "c_key_normalization_variants",
        all_passed,
        "; ".join(details) if details else "All variants extracted",
    )
    return all_passed


# ---------------------------------------------------------------------------
# Bug 1(d) — Hypothesis PBT for key normalization
# ---------------------------------------------------------------------------

# Strategy: generate random key format variants for the three score fields
_SCORE_BASES = ["profile_usage", "task_usage", "integration"]


def _random_key_variant(base: str) -> st.SearchStrategy[str]:
    """Generate random key format variants for a score base name."""
    return st.one_of(
        # snake_case with _score suffix
        st.just(f"{base}_score"),
        # snake_case without suffix
        st.just(base),
        # Space-separated with Score suffix
        st.just(f"{base.replace('_', ' ')} score"),
        # Title Case with Score
        st.just(f"{base.replace('_', ' ').title()} Score"),
        # Title Case without Score
        st.just(f"{base.replace('_', ' ').title()}"),
        # CamelCase with Score
        st.just("".join(w.capitalize() for w in base.split("_")) + "Score"),
        # camelCase with Score (first word lowercase)
        st.just(
            base.split("_")[0]
            + "".join(w.capitalize() for w in base.split("_")[1:])
            + "Score"
        ),
        # Hyphenated with -score
        st.just(f"{base.replace('_', '-')}-score"),
        # UPPER_SNAKE_CASE
        st.just(f"{base.upper()}_SCORE"),
        # Random casing of the base + _score
        st.just(base.upper() + "_score"),
    )


@given(
    profile_key=_random_key_variant("profile_usage"),
    task_key=_random_key_variant("task_usage"),
    integration_key=_random_key_variant("integration"),
    profile_val=st.integers(min_value=1, max_value=5),
    task_val=st.integers(min_value=1, max_value=5),
    integration_val=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=200, derandomize=True)
def test_d_hypothesis_key_normalization(
    profile_key, task_key, integration_key,
    profile_val, task_val, integration_val,
):
    """(d) Hypothesis PBT for key normalization.

    **Validates: Requirements 1.3**

    For any reasonable key format variant, _normalize_judge_keys should
    extract all three canonical scores. On unfixed code, many variants fail.
    """
    data = {
        profile_key: profile_val,
        task_key: task_val,
        integration_key: integration_val,
    }
    normalized = _normalize_judge_keys(data)
    assert normalized.get("profile_usage_score") is not None, (
        f"profile_usage_score is None for key '{profile_key}' in {data}"
    )
    assert normalized.get("task_usage_score") is not None, (
        f"task_usage_score is None for key '{task_key}' in {data}"
    )
    assert normalized.get("integration_score") is not None, (
        f"integration_score is None for key '{integration_key}' in {data}"
    )


# ---------------------------------------------------------------------------
# Bug 2 — Audit Query
# ---------------------------------------------------------------------------

def test_e_missing_diagnostics_injection_status():
    """(e) Missing diagnostics + failed audit produces unknown status.

    **Validates: Requirements 1.5, 1.6**

    When diagnostics are absent and audit returns 0, the RetrievalLog should
    have an injection_status field equal to "unknown". On unfixed code, the
    RetrievalLog model has no injection_status field — this WILL FAIL.
    """
    log = RetrievalLog(shared_memory_count=0)

    has_field = hasattr(log, "injection_status")
    correct_value = False
    if has_field:
        correct_value = log.injection_status == "unknown"

    # We expect the field to exist AND have value "unknown" for a zero-count log.
    # On unfixed code, the field doesn't exist at all.
    passed = has_field and correct_value

    record(
        "e_missing_diagnostics_injection_status",
        passed,
        f"has injection_status field: {has_field}"
        + (f", value: {log.injection_status}" if has_field else ""),
    )
    return passed


def test_f_retrieval_log_lacks_injection_status_field():
    """(f) RetrievalLog does NOT have injection_status field (confirms model gap).

    **Validates: Requirements 1.6**

    This test verifies that RetrievalLog currently lacks the injection_status
    field, confirming the observability gap. After the fix adds the field,
    this test WILL FAIL (which is correct — the gap is closed).
    """
    log = RetrievalLog()
    fields = set(RetrievalLog.model_fields.keys())

    # On unfixed code, injection_status is NOT in the model fields.
    # After the fix, it WILL be present, so this assertion will fail.
    lacks_field = "injection_status" not in fields

    # We want to assert the field IS present (expected behavior after fix).
    # So we assert NOT lacks_field. On unfixed code this FAILS.
    field_present = not lacks_field

    record(
        "f_retrieval_log_injection_status_field",
        field_present,
        f"Model fields: {sorted(fields)}; injection_status present: {field_present}",
    )
    return field_present


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all():
    """Run all bug condition exploration tests and report results."""
    print("=" * 70)
    print("Bug Condition Exploration Tests")
    print("These tests are EXPECTED TO FAIL on unfixed code.")
    print("Failure confirms the bugs exist.")
    print("=" * 70)

    # Bug 1 — Judge Scoring
    print("\n--- Bug 1: Judge Scoring ---")
    test_a_judge_prompt_lacks_injected_context_awareness()
    test_b_generic_penalty_misapplication()
    test_c_key_normalization_variant_handling()

    # Bug 1(d) — Hypothesis PBT
    print("\n--- Bug 1(d): Hypothesis PBT for key normalization ---")
    try:
        test_d_hypothesis_key_normalization()
        record(
            "d_hypothesis_key_normalization",
            True,
            "Hypothesis test passed unexpectedly — all variants handled",
        )
    except AssertionError:
        record(
            "d_hypothesis_key_normalization",
            False,
            "Hypothesis found counterexample (AssertionError)",
        )
    except Exception as e:
        record(
            "d_hypothesis_key_normalization",
            False,
            f"Hypothesis found counterexample: {e}",
        )

    # Bug 2 — Audit Query
    print("\n--- Bug 2: Audit Query ---")
    test_e_missing_diagnostics_injection_status()
    test_f_retrieval_log_lacks_injection_status_field()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    expected_fails = sum(1 for _, s, _ in results if s == "EXPECTED FAIL")
    unexpected_passes = sum(1 for _, s, _ in results if s == "UNEXPECTED PASS")
    print(f"  Expected Fails:    {expected_fails}")
    print(f"  Unexpected Passes: {unexpected_passes}")
    print(f"  Total tests:       {len(results)}")

    if expected_fails == len(results):
        print("\nAll tests failed as expected — bugs confirmed.")
    elif unexpected_passes > 0:
        print(f"\nWARNING: {unexpected_passes} test(s) passed unexpectedly!")
    print("=" * 70)


if __name__ == "__main__":
    run_all()
