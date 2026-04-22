"""Preservation property tests for the benchmark harness.

These tests capture baseline behavior that MUST remain unchanged after
the bugfix. They MUST PASS on the current (unfixed) code.

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
"""

import sys
sys.path.insert(0, ".")

import traceback

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import integers

from benchmarks.shared_memory.judge import (
    LLMJudge,
    _clamp_score,
    _normalize_judge_keys,
)
from benchmarks.shared_memory.models import (
    InjectedMemoryEntry,
    InjectionDiagnostics,
    RetrievalLog,
    RetrievalLogEntry,
    SyntheticProfile,
    SyntheticTaskContext,
)
from benchmarks.shared_memory.pipeline import AgentPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

results = []


def record(name: str, passed: bool, detail: str = ""):
    """Record a test result."""
    status = "PASS" if passed else "FAIL"
    results.append((name, status, detail))
    print(f"  [{status}] {name}")
    if detail:
        print(f"    Detail: {detail}")


# ---------------------------------------------------------------------------
# (a) _clamp_score preservation
# ---------------------------------------------------------------------------

@given(value=integers())
@settings(max_examples=100)
def test_a_clamp_score_preservation(value):
    """(a) _clamp_score clamps to [1, 5] for all integers.

    **Validates: Requirements 3.3**

    Values < 1 become 1, values > 5 become 5, values in [1, 5] unchanged.
    """
    result = _clamp_score(value, "test_score")
    assert 1 <= result <= 5, f"_clamp_score({value}) = {result}, not in [1, 5]"
    if value < 1:
        assert result == 1, f"_clamp_score({value}) = {result}, expected 1"
    elif value > 5:
        assert result == 5, f"_clamp_score({value}) = {result}, expected 5"
    else:
        assert result == value, f"_clamp_score({value}) = {result}, expected {value}"


# ---------------------------------------------------------------------------
# (b) _retrieval_log_from_diagnostics preservation
# ---------------------------------------------------------------------------

# Strategy for generating InjectedMemoryEntry objects
_injected_memory_entry_strategy = st.builds(
    InjectedMemoryEntry,
    owner_agent=st.sampled_from([
        "profile_agent", "task_agent", "assistant_agent", "other_agent",
    ]),
    memory_type=st.sampled_from(["profile", "task_context", "preference", "note"]),
    match_score=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
)


@given(
    entries=st.lists(_injected_memory_entry_strategy, min_size=1, max_size=10),
)
@settings(max_examples=100)
def test_b_retrieval_log_from_diagnostics_preservation(entries):
    """(b) _retrieval_log_from_diagnostics builds correct RetrievalLog.

    **Validates: Requirements 3.4**

    For any InjectionDiagnostics with injected_count > 0, the function
    builds a RetrievalLog with:
    - shared_memory_count == diagnostics.injected_count
    - correct number of RetrievalLogEntry items
    - cross_agent_found is True iff any owner_agent != "assistant_agent"
    """
    diagnostics = InjectionDiagnostics(
        injected_count=len(entries),
        injected_memories=entries,
    )

    pipeline = AgentPipeline(share_memory=True)
    log = pipeline._retrieval_log_from_diagnostics(diagnostics)

    assert isinstance(log, RetrievalLog), "Result is not a RetrievalLog"
    assert log.shared_memory_count == diagnostics.injected_count, (
        f"shared_memory_count={log.shared_memory_count}, "
        f"expected={diagnostics.injected_count}"
    )
    assert len(log.retrieved_memories) == len(entries), (
        f"retrieved_memories count={len(log.retrieved_memories)}, "
        f"expected={len(entries)}"
    )

    # Verify cross_agent_found
    expected_cross_agent = any(
        e.owner_agent != "assistant_agent" for e in entries
    )
    assert log.cross_agent_found == expected_cross_agent, (
        f"cross_agent_found={log.cross_agent_found}, "
        f"expected={expected_cross_agent}"
    )

    # Verify each entry maps correctly
    for i, (orig, mapped) in enumerate(zip(entries, log.retrieved_memories)):
        assert mapped.owner_agent == orig.owner_agent, (
            f"Entry {i}: owner_agent={mapped.owner_agent}, "
            f"expected={orig.owner_agent}"
        )
        assert mapped.memory_type == orig.memory_type, (
            f"Entry {i}: memory_type={mapped.memory_type}, "
            f"expected={orig.memory_type}"
        )


# ---------------------------------------------------------------------------
# (c) _build_retrieval_log_from_search preservation
# ---------------------------------------------------------------------------

# Strategy for generating search result dicts with varying structures
_search_result_strategy = st.one_of(
    # Empty / missing keys
    st.just({}),
    st.just({"response": {}}),
    st.just({"response": {"search_results": []}}),
    st.just({"response": {"search_results": None}}),
    # Non-dict response
    st.just({"response": "not a dict"}),
    st.just(None),
    st.just("not a dict"),
    # Valid entries with metadata
    st.builds(
        lambda entries: {
            "response": {
                "search_results": entries,
            }
        },
        entries=st.lists(
            st.fixed_dictionaries({
                "metadata": st.fixed_dictionaries({
                    "owner_agent": st.sampled_from([
                        "profile_agent", "task_agent", "assistant_agent",
                    ]),
                    "memory_type": st.sampled_from([
                        "profile", "task_context", "preference",
                    ]),
                    "sharing_policy": st.sampled_from(["shared", "private"]),
                }),
            }),
            min_size=0,
            max_size=5,
        ),
    ),
    # Entries with missing metadata
    st.just({"response": {"search_results": [{"metadata": {}}]}}),
    st.just({"response": {"search_results": [{"metadata": None}]}}),
    st.just({"response": {"search_results": [{}]}}),
)


@given(search_result=_search_result_strategy)
@settings(max_examples=100)
def test_c_build_retrieval_log_from_search_preservation(search_result):
    """(c) _build_retrieval_log_from_search never crashes.

    **Validates: Requirements 3.5**

    For any search response dict (including malformed ones), the function
    never crashes and always returns a RetrievalLog.
    """
    pipeline = AgentPipeline(share_memory=True)
    log = pipeline._build_retrieval_log_from_search(search_result)
    assert isinstance(log, RetrievalLog), (
        f"Result is not a RetrievalLog: {type(log)}"
    )


# ---------------------------------------------------------------------------
# (d) Phase 1 rubric structure preservation
# ---------------------------------------------------------------------------

def test_d_phase1_rubric_structure_preservation():
    """(d) _build_judge_prompt preserves rubric structure.

    **Validates: Requirements 3.1, 3.2**

    The judge prompt for a generic response must contain the scoring rubric
    with Profile Usage, Task Usage, and Integration sections.
    """
    judge = LLMJudge()
    profile = SyntheticProfile(
        user_name="TestUser",
        preferred_tools=["vim"],
        preferred_language="Go",
        response_style="detailed",
    )
    task_context = SyntheticTaskContext(
        current_project="web server",
        active_experiment="load testing",
        goals=["handle 10k concurrent connections"],
        blockers=["high tail latency"],
        next_steps=["add connection pooling"],
    )

    query = "How can I improve performance?"
    # A generic response that doesn't reference profile/task specifics
    response = (
        "You should consider optimizing your code and using better algorithms. "
        "Performance tuning is important for any project. Make sure to profile "
        "your application and identify bottlenecks before making changes."
    )

    messages = judge._build_judge_prompt(query, response, profile, task_context)
    full_prompt = " ".join(m["content"] for m in messages)

    has_profile_usage = "Profile Usage" in full_prompt
    has_task_usage = "Task Usage" in full_prompt
    has_integration = "Integration" in full_prompt

    all_present = has_profile_usage and has_task_usage and has_integration

    record(
        "d_phase1_rubric_structure",
        all_present,
        f"Profile Usage: {has_profile_usage}, "
        f"Task Usage: {has_task_usage}, "
        f"Integration: {has_integration}",
    )
    return all_present


# ---------------------------------------------------------------------------
# (e) _normalize_judge_keys preservation for canonical keys
# ---------------------------------------------------------------------------

@given(
    profile_val=integers(),
    task_val=integers(),
    integration_val=integers(),
)
@settings(max_examples=100)
def test_e_normalize_judge_keys_canonical_preservation(
    profile_val, task_val, integration_val,
):
    """(e) _normalize_judge_keys returns canonical keys unchanged.

    **Validates: Requirements 3.3**

    When given exact canonical key names, all three scores are returned
    unchanged. This must pass on unfixed code.
    """
    data = {
        "profile_usage_score": profile_val,
        "task_usage_score": task_val,
        "integration_score": integration_val,
    }
    normalized = _normalize_judge_keys(data)

    assert normalized.get("profile_usage_score") == profile_val, (
        f"profile_usage_score: got {normalized.get('profile_usage_score')}, "
        f"expected {profile_val}"
    )
    assert normalized.get("task_usage_score") == task_val, (
        f"task_usage_score: got {normalized.get('task_usage_score')}, "
        f"expected {task_val}"
    )
    assert normalized.get("integration_score") == integration_val, (
        f"integration_score: got {normalized.get('integration_score')}, "
        f"expected {integration_val}"
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all():
    """Run all preservation property tests and report results."""
    print("=" * 70)
    print("Preservation Property Tests")
    print("These tests MUST PASS on unfixed code.")
    print("=" * 70)

    # (a) _clamp_score preservation (Hypothesis PBT)
    print("\n--- (a) _clamp_score preservation ---")
    try:
        test_a_clamp_score_preservation()
        record("a_clamp_score_preservation", True, "All integers clamped correctly")
    except Exception as e:
        record("a_clamp_score_preservation", False, f"Failed: {e}")

    # (b) _retrieval_log_from_diagnostics preservation (Hypothesis PBT)
    print("\n--- (b) _retrieval_log_from_diagnostics preservation ---")
    try:
        test_b_retrieval_log_from_diagnostics_preservation()
        record(
            "b_retrieval_log_from_diagnostics_preservation",
            True,
            "All diagnostics correctly mapped to RetrievalLog",
        )
    except Exception as e:
        record(
            "b_retrieval_log_from_diagnostics_preservation",
            False,
            f"Failed: {e}",
        )

    # (c) _build_retrieval_log_from_search preservation (Hypothesis PBT)
    print("\n--- (c) _build_retrieval_log_from_search preservation ---")
    try:
        test_c_build_retrieval_log_from_search_preservation()
        record(
            "c_build_retrieval_log_from_search_preservation",
            True,
            "All search results handled without crash",
        )
    except Exception as e:
        record(
            "c_build_retrieval_log_from_search_preservation",
            False,
            f"Failed: {e}",
        )

    # (d) Phase 1 rubric structure preservation (unit test)
    print("\n--- (d) Phase 1 rubric structure preservation ---")
    test_d_phase1_rubric_structure_preservation()

    # (e) _normalize_judge_keys canonical preservation (Hypothesis PBT)
    print("\n--- (e) _normalize_judge_keys canonical preservation ---")
    try:
        test_e_normalize_judge_keys_canonical_preservation()
        record(
            "e_normalize_judge_keys_canonical_preservation",
            True,
            "All canonical keys returned unchanged",
        )
    except Exception as e:
        record(
            "e_normalize_judge_keys_canonical_preservation",
            False,
            f"Failed: {e}",
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total:  {len(results)}")

    if failed == 0:
        print("\nAll preservation tests PASSED — baseline behavior confirmed.")
    else:
        print(f"\nWARNING: {failed} test(s) FAILED — baseline behavior broken!")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
