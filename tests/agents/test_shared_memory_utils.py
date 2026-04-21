"""Unit tests for shared_memory_utils edge cases.

Validates: Requirements 1.2, 1.5
"""

import sys
sys.path.insert(0, ".")

from cerebrum.example.agents.shared_memory_utils import (
    build_memory_metadata,
    POLICY_PRIVATE,
    FIELD_SHARING_POLICY,
    FIELD_OWNER_AGENT,
    FIELD_USER_ID,
    FIELD_MEMORY_TYPE,
)


def test_sharing_policy_defaults_to_private():
    """sharing_policy defaults to 'private' when omitted. (Req 1.2)"""
    result = build_memory_metadata(
        owner_agent="agent_a",
        user_id="user_1",
        memory_type="profile",
    )
    assert result[FIELD_SHARING_POLICY] == POLICY_PRIVATE, (
        f"Expected sharing_policy='private', got {result[FIELD_SHARING_POLICY]!r}"
    )
    print("PASSED: sharing_policy defaults to 'private'")


def test_empty_owner_agent_raises():
    """ValueError raised for owner_agent=''. (Req 1.5)"""
    try:
        build_memory_metadata(
            owner_agent="",
            user_id="user_1",
            memory_type="profile",
        )
        assert False, "Expected ValueError for empty owner_agent"
    except ValueError as exc:
        assert "owner_agent" in str(exc), (
            f"Error message should mention 'owner_agent', got: {exc}"
        )
    print("PASSED: ValueError for owner_agent=''")


def test_empty_user_id_raises():
    """ValueError raised for user_id=''. (Req 1.5)"""
    try:
        build_memory_metadata(
            owner_agent="agent_a",
            user_id="",
            memory_type="profile",
        )
        assert False, "Expected ValueError for empty user_id"
    except ValueError as exc:
        assert "user_id" in str(exc), (
            f"Error message should mention 'user_id', got: {exc}"
        )
    print("PASSED: ValueError for user_id=''")


def test_extra_kwargs_passed_through():
    """Extra keyword arguments are included in the returned metadata."""
    result = build_memory_metadata(
        owner_agent="agent_a",
        user_id="user_1",
        memory_type="conversation",
        sharing_policy="shared",
        agent_id="mem0_abc",
        priority=5,
    )
    assert result["agent_id"] == "mem0_abc", (
        f"Expected agent_id='mem0_abc', got {result.get('agent_id')!r}"
    )
    assert result["priority"] == 5, (
        f"Expected priority=5, got {result.get('priority')!r}"
    )
    # Verify standard fields are still present
    assert result[FIELD_OWNER_AGENT] == "agent_a"
    assert result[FIELD_USER_ID] == "user_1"
    assert result[FIELD_MEMORY_TYPE] == "conversation"
    assert result[FIELD_SHARING_POLICY] == "shared"
    print("PASSED: extra kwargs passed through correctly")


if __name__ == "__main__":
    test_sharing_policy_defaults_to_private()
    test_empty_owner_agent_raises()
    test_empty_user_id_raises()
    test_extra_kwargs_passed_through()
    print("\nAll tests passed.")
